"""Reduce a menu to a tagged family of modules and everything needed to run them.

Given a full cmsDriver configuration, keep only the modules whose C++ type carries
one of a set of tags and that the configuration actually schedules, plus the
smallest set of other modules, Paths and ESProducers needed to feed them, and drop
everything else.  The tags default to '@alpaka' alone, which selects the
heterogeneous modules, but nothing in the machinery depends on that value.

Two rules define the result:
- seed rule: a module is a seed if one of the tags appears in its type and the
             module is scheduled in the input configuration. Testing against the schedule
             rather than against the set of defined modules is what makes the
             result follow the eras and process modifiers of the input.

- stop rule: a dependency on something this process does not schedule refers to a
             product that comes from the input file, so the closure stops there.

The seeds are wrapped in the fixed HLT structure (HLTBeginSequence and the L1
accept filter before them, HLTEndSequence after them) and their dependencies are
put in a Task associated with the Path, so that the framework runs them on demand
and in the right order, and any EDFilter among them has no say on the Path
decision. The Schedule is the new Path followed by the two HLT bookkeeping paths
(HLTriggerFinalPath and HLTAnalyzerEndpath)

Dependencies come from the framework itself, via the Tracer service, which reports
what every module declared it consumes once the job has resolved its lookups.
Running a short cmsRun job is therefore not optional, and the configuration has
to be runnable. A walk over the InputTags is used alongside the tracer, to name
the dependency in terms of the parameter it was found in.
"""

import collections
import importlib
import os
import re
import subprocess
import sys
import tempfile

import FWCore.ParameterSet.Config as cms

# A module is a seed when one of these appears in its C++ type. Nothing else about
# the machinery is specific to heterogeneous modules: any word, or set of words, a
# menu uses to mark a family of modules works the same way.
DEFAULT_MODULE_TAGS = ("@alpaka",)


def hasTag(component, moduleTags):
    """Whether a module or an EventSetup component carries one of the tags.

    A tag is looked for anywhere in the C++ type, not only at its end.  The default
    '@alpaka' is a suffix the framework appends, so it selects exactly the
    heterogeneous modules either way, while a word such as 'Portable' is found
    wherever a type happens to carry it.  Several tags select the union of the
    families they name.
    """
    return any(tag in component.type_() for tag in moduleTags)


def _cfiObject(cfi):
    """The name of the object a cfi provides, its file name without the _cfi suffix."""
    name = cfi.rsplit(".", 1)[-1]
    return name[:-len("_cfi")] if name.endswith("_cfi") else name


def _cfiPackage(cfi):
    """The menu package a cfi belongs to, dropping the subpackage and the file name."""
    return cfi.rsplit(".", 2)[0]


# The fixed HLT structure the seeds are wrapped in. Only the cfi of the L1 accept
# filter is spelled out: the filter label is the object that cfi provides, and the
# package holding it is the menu whose sequences the begin and end sequences are
# taken from.
HLT_L1_FILTER_CFI = "HLTrigger.Configuration.HLT_75e33.modules.hltL1GTAcceptFilter_cfi"
HLT_L1_FILTER = _cfiObject(HLT_L1_FILTER_CFI)
HLT_MENU = _cfiPackage(HLT_L1_FILTER_CFI)
HLT_BEGIN_SEQUENCE = "HLTBeginSequence"
HLT_END_SEQUENCE = "HLTEndSequence"
HLT_FINAL_PATHS = ("HLTriggerFinalPath", "HLTAnalyzerEndpath")


def collectInputTags(pset, base, out):
    """Append (InputTag, parameterPath) for every InputTag and VInputTag under pset.

    Descends into PSets, VPSets and VInputTags the same way that
    MassSearchReplaceAnyInputTagVisitor does, but collects instead of replacing.
    """
    if not isinstance(pset, cms._Parameterizable):
        return
    for name in pset.parameterNames_():
        value = getattr(pset, name)
        parameter = "%s.%s" % (base, name)
        if isinstance(value, cms.PSet):
            collectInputTags(value, parameter, out)
        elif not hasattr(value, "isCompatibleCMSType"):
            continue
        elif value.isCompatibleCMSType(cms.VPSet):
            for i, entry in enumerate(value):
                collectInputTags(entry, "%s[%d]" % (parameter, i), out)
        elif value.isCompatibleCMSType(cms.VInputTag) and value:
            for i, tag in enumerate(value):
                # a VInputTag may be declared as a list of plain strings
                if not isinstance(tag, cms.InputTag):
                    tag = cms.InputTag(tag)
                out.append((tag, "%s[%d]" % (parameter, i)))
        elif value.isCompatibleCMSType(cms.InputTag) and value:
            out.append((value, parameter))


def scheduledLabels(process):
    """The labels of the modules the framework can run in this job.

    A module counts as scheduled if it is on a Path or EndPath, or in a Task
    associated with one of them or with the Schedule: modules on Tasks are run on
    demand, so they are just as available to a consumer as scheduled ones.
    """
    schedule = process.schedule_()
    if schedule is None:
        schedule = cms.Schedule(*(list(process.paths_().values()) +
                                  list(process.endpaths_().values())))
    return set(schedule.moduleNames())


# Tracer service dump
_ED_SECTION = "All modules and modules in the current process whose products they consume:"
_ES_SECTION = "All EventSetup modules:"
_END_SECTION = "All modules (listed by class and label)"

_TRACER_ENTRY = re.compile(r"^(?P<indent> +)(?P<type>.+?)/'(?P<label>[^']*)'")

_GRAPH_JOB_TEMPLATE = """\
import sys
sys.path.insert(0, %(directory)r)

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.processFromFile import processFromFile

process = processFromFile(%(filename)r, %(configArgs)r)

process.Tracer = cms.Service(
    "Tracer",
    dumpPathsAndConsumes=cms.untracked.bool(True),
    dumpContextForLabels=cms.untracked.vstring(),
)
process.maxEvents.input = 0
"""


class FrameworkGraph(object):
    """What the framework reported that every module consumes.

    edToEd    consumer label -> set of EDProducer/EDFilter labels, including the
              PathStatusInserter modules that carry a Path label
    edToEs    consumer label -> set of EventSetup module labels
    esToEs    EventSetup module label -> set of EventSetup module labels
    """

    def __init__(self, edToEd=None, edToEs=None, esToEs=None):
        self.edToEd = edToEd or {}
        self.edToEs = edToEs or {}
        self.esToEs = esToEs or {}

    def eventSetupClosure(self, labels):
        """Every EventSetup module reachable from the given ED modules."""
        kept = set()
        queue = collections.deque()
        for label in labels:
            for target in self.edToEs.get(label, ()):
                if target not in kept:
                    kept.add(target)
                    queue.append(target)
        while queue:
            for target in self.esToEs.get(queue.popleft(), ()):
                if target not in kept:
                    kept.add(target)
                    queue.append(target)
        return kept


def parseTracerLog(filename):
    """Read the dumpPathsAndConsumes output of the Tracer service."""
    graph = FrameworkGraph()
    section = None
    current = None
    consumed = None
    with open(filename, errors="replace") as log:
        for line in log:
            if line.startswith(_ED_SECTION):
                section, current = "ed", None
                continue
            if line.startswith(_ES_SECTION):
                section, current = "es", None
                continue
            if line.startswith(_END_SECTION):
                section = None
                continue
            if section is None:
                continue
            entry = _TRACER_ENTRY.match(line.rstrip("\n"))
            if entry is None:
                stripped = line.strip()
                if stripped.startswith("consumes products from these EventSetup modules:"):
                    consumed = "es"
                elif "from these EventSetup modules:" in stripped:
                    consumed = "es"
                elif stripped.startswith("consumes products from these modules:"):
                    consumed = "ed"
                continue
            indent, label = len(entry.group("indent")), entry.group("label")
            # An EventSetup module need not have a label: the EventSetup section
            # prints its type in place of the missing label, the module section
            # prints the empty label. Use the type in both, so that the two
            # sections name the same module in the same way.
            esName = label or entry.group("type").strip()
            if indent <= 2:
                current = esName if section == "es" else label
                consumed = None
            elif current is not None and consumed is not None:
                if section == "es":
                    graph.esToEs.setdefault(current, set()).add(esName)
                elif consumed == "es":
                    graph.edToEs.setdefault(current, set()).add(esName)
                else:
                    graph.edToEd.setdefault(current, set()).add(label)
    if not graph.edToEd and not graph.esToEs:
        raise RuntimeError("no dependency information found in %s; the Tracer "
                           "service did not report as expected" % filename)
    return graph


def buildFrameworkGraph(filename, configArgs=(), directory=None):
    """Run a short cmsRun job on filename and return the path of its Tracer log.

    The job processes no events: the Tracer reports at the end of the framework's
    lookup initialisation, so the information is complete as soon as the job has
    started.
    """
    if directory is None:
        directory = tempfile.mkdtemp(prefix="hltDumpTaggedModules-")
    log = os.path.join(directory, "tracer.log")
    job = os.path.join(directory, "tracer_cfg.py")
    with open(job, "w") as script:
        script.write(_GRAPH_JOB_TEMPLATE % {
            "directory": os.path.dirname(os.path.abspath(filename)) or ".",
            "filename": os.path.abspath(filename),
            "configArgs": list(configArgs),
        })
    print("Running cmsRun to collect the framework's dependency information...",
          file=sys.stderr)
    with open(log, "w") as output:
        result = subprocess.run(["cmsRun", job], stdout=output,
                                stderr=subprocess.STDOUT, universal_newlines=True)
    if result.returncode != 0:
        with open(log, errors="replace") as output:
            tail = output.read()[-4000:]
        raise RuntimeError("the dependency job failed:\n%s" % tail)
    print("Dependency information kept in %s" % log, file=sys.stderr)
    return log


class ModuleGraph(object):
    """The data flow graph of the EDProducers and EDFilters of a Process."""

    def __init__(self, process, frameworkGraph=None, moduleTags=DEFAULT_MODULE_TAGS):
        self.process = process
        self.moduleTags = moduleTags
        self.scheduled = scheduledLabels(process)
        self.modules = {}
        self.modules.update(process.producers_())
        self.modules.update(process.filters_())
        self.framework = frameworkGraph or FrameworkGraph()
        self._aliases = process.aliases_()
        self._eventSetup = set(process.es_producers_()) | set(process.es_sources_())
        self._paths = set(process.paths_()) | set(process.endpaths_())
        self._tags = {}
        for label, module in self.modules.items():
            tags = []
            collectInputTags(module, label, tags)
            self._tags[label] = tags

    def taggedModules(self):
        """All EDProducers and EDFilters of the process whose type carries the tag."""
        return sorted(label for label, module in self.modules.items()
                      if hasTag(module, self.moduleTags))

    def seeds(self):
        """The tagged modules this configuration schedules."""
        return [label for label in self.taggedModules() if label in self.scheduled]

    def _resolveAlias(self, label):
        """The parameter names of an EDAlias are the labels it aliases."""
        if label in self._aliases:
            return list(self._aliases[label].parameterNames_())
        return [label]

    def dependencies(self, label):
        """[(label, why)] for everything this module consumes."""
        result = collections.OrderedDict()
        for tag, parameter in self._tags.get(label, []):
            moduleLabel = tag.getModuleLabel()
            if not moduleLabel:
                continue
            for target in self._resolveAlias(moduleLabel):
                result.setdefault(target, parameter)
        for target in sorted(self.framework.edToEd.get(label, ())):
            result.setdefault(target, "consumes, reported by the framework")
        return list(result.items())

    def classify(self, label):
        """Why the closure does or does not follow into label."""
        if label in self.modules and label in self.scheduled:
            return "module"
        if label in self._paths:
            return "path"
        if label in self._eventSetup:
            return "eventsetup"
        if label not in self.modules:
            return "not-in-process"
        return "defined-not-scheduled"

    def closure(self, seeds):
        """(provenance, boundary, paths) for the upstream closure of seeds.

        provenance maps every kept label to (consumer, why), the edge the walk
        first reached it through, or to None for a seed.
        boundary maps every label the walk stopped at to why it stopped
        and who wanted it.
        paths holds the Paths that have to stay in the Schedule because a module
        consumes their edm::HLTPathStatus, which a Path only publishes when it is
        scheduled. Their modules join the closure as well.
        """
        provenance = collections.OrderedDict((label, None) for label in seeds)
        boundary = {}
        paths = {}
        queue = collections.deque(seeds)
        while queue:
            current = queue.popleft()
            for target, why in self.dependencies(current):
                if target in provenance:
                    continue
                kind = self.classify(target)
                if kind == "module":
                    provenance[target] = (current, why)
                    queue.append(target)
                elif kind == "path":
                    if target not in paths:
                        paths[target] = (current, why)
                        for label in getattr(self.process, target).moduleNames():
                            if label not in provenance:
                                provenance[label] = (target, "on Path %s" % target)
                                queue.append(label)
                else:
                    entry = boundary.setdefault(target, {"kind": kind, "consumers": []})
                    entry["consumers"].append((current, why))
        return provenance, boundary, paths

    def topologicalOrder(self, labels):
        """labels ordered so that a module comes after the ones it consumes from."""
        remaining = set(labels)
        order = []
        state = {}

        def visit(label):
            if state.get(label) is not None:
                return
            state[label] = "visiting"
            for target, _ in sorted(self.dependencies(label)):
                if target in remaining:
                    visit(target)
            state[label] = "done"
            order.append(label)

        for label in sorted(remaining):
            visit(label)
        return order


def _modifierNames(process):
    """Best effort names of the Modifiers and ModifierChains applied to a Process."""
    modifiers = getattr(process, "_Process__modifiers", ())
    if not modifiers:
        return []
    names = dict((id(modifier), None) for modifier in modifiers)
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith("Configuration."):
            continue
        for attribute, value in getattr(module, "__dict__", {}).items():
            if id(value) in names and names[id(value)] is None:
                names[id(value)] = attribute
    return [names[id(modifier)] or type(modifier).__name__ for modifier in modifiers]


def moduleNamesOf(item):
    """The module labels of a Sequence or Path, or of a single module."""
    if hasattr(item, "moduleNames"):
        return set(item.moduleNames())
    return set([item.label_()])


def _fromMenu(process, name, subpackage):
    """The object called name, taken from the process or loaded from the menu.

    Returns None when the process does not have it and the menu does not provide
    it either, so that the reduction still works on a process that is not an HLT
    menu.
    """
    if not hasattr(process, name):
        try:
            process.load("%s.%s.%s_cfi" % (HLT_MENU, subpackage, name))
        except ImportError:
            return None
    return getattr(process, name, None)


def hltStructure(process):
    """(prologue, epilogue, finalPaths) - the fixed HLT structure around the seeds.

    The Path begins with HLTBeginSequence followed by the L1 accept filter and
    ends with HLTEndSequence, and the Schedule keeps the two HLT bookkeeping
    paths.  Whatever the process does not already have is loaded from the menu,
    since the L1 accept filter in particular is not part of the menu's own paths.
    """
    prologue = [item for item in (_fromMenu(process, HLT_BEGIN_SEQUENCE, "sequences"),
                                  _fromMenu(process, HLT_L1_FILTER, "modules"))
                if item is not None]
    epilogue = [item for item in [_fromMenu(process, HLT_END_SEQUENCE, "sequences")]
                if item is not None]
    finalPaths = [getattr(process, name) for name in HLT_FINAL_PATHS
                  if hasattr(process, name)]
    return prologue, epilogue, finalPaths


def pruneEventSetup(process, graph, keptModules, moduleTags=DEFAULT_MODULE_TAGS):
    """Drop the EventSetup modules none of the kept modules can reach.

    ESSources are dropped on the same rule as ESProducers, because an EmptyESSource
    whose Record nothing in the job consumes is a fatal configuration error.
    Tagged EventSetup modules are always kept: for @alpaka, which backend flavour
    of one of them is picked depends on the accelerator of the job that produced
    the graph, while the reduced configuration is meant to run on any backend.
    """
    needed = graph.eventSetupClosure(keptModules)
    dropped = []
    for components in (process.es_producers_(), process.es_sources_()):
        for label, component in sorted(components.items()):
            # an unlabelled EventSetup module is known by its type instead
            if (label in needed or component.type_() in needed
                    or hasTag(component, moduleTags)):
                continue
            dropped.append(label)
            delattr(process, label)
    # an ESPrefer resolves its target by type, so it has to go as soon as no
    # EventSetup module of that type is left
    remaining = set()
    for components in (process.es_producers_(), process.es_sources_()):
        remaining |= set(component.type_() for component in components.values())
    for label, prefer in sorted(process.es_prefers_().items()):
        if prefer.type_() not in remaining:
            dropped.append(label)
            delattr(process, label)
    return sorted(dropped)


def reduceToTaggedModules(process, frameworkGraph, pathName="DST_HeterogeneousReco",
                          moduleTags=DEFAULT_MODULE_TAGS):
    """Keep only the seeds and everything needed to run them; return a manifest.

    The process is modified in place: a new Path holding the HLT prologue, the
    seeds and the HLT epilogue, and a Task holding their dependencies, replace the
    original Paths and EndPaths; the Schedule becomes the Paths whose status is
    consumed, the new Path, and the HLT bookkeeping paths; unreachable ESProducers
    are dropped; and the process is pruned.
    """
    taskName = pathName + "Task"
    for name in (pathName, taskName):
        if hasattr(process, name):
            raise RuntimeError("the process already has an attribute named '%s'" % name)

    # Resolved before the graph is built, so that a filter imported here is part
    # of the process and has its dependencies walked like any other module.
    prologue, epilogue, finalPaths = hltStructure(process)

    graph = ModuleGraph(process, frameworkGraph, moduleTags)
    seeds = graph.seeds()
    if not seeds:
        raise RuntimeError("this configuration schedules no module tagged %s, "
                           "there is nothing to keep" % ", ".join(moduleTags))

    wrapper = set()
    for item in prologue + epilogue:
        wrapper |= moduleNamesOf(item)
    finalPathModules = set()
    for item in finalPaths:
        finalPathModules |= moduleNamesOf(item)

    # The prologue and the epilogue are scheduled whatever the seeds are, so their
    # dependencies belong to the closure.  The HLT bookkeeping paths are not
    # seeded: they summarise the whole menu, so their closure would be the whole
    # menu.  They keep their modules by being in the Schedule, and accept that the
    # products they summarise are no longer produced.
    provenance, boundary, requiredPaths = graph.closure(seeds + sorted(wrapper - set(seeds)))

    # keep the required Paths in their original Schedule order, before the new
    # Path, so that their HLTPathStatus is available when it is consumed
    scheduleOrder = [item.label_() for item in (process.schedule_() or [])]
    keptPaths = ([name for name in scheduleOrder if name in requiredPaths] +
                 sorted(name for name in requiredPaths if name not in scheduleOrder))
    keptPathModules = set()
    for name in keptPaths:
        keptPathModules |= set(getattr(process, name).moduleNames())

    # An @alpaka EDFilter goes first, so that it gates the rest of the Path; the
    # producers follow in topological order, because modules on a Path run where
    # they are listed and may not consume from a module listed after them.
    # Dependencies live on the Task, so they run on demand whatever their position.
    filterSeeds = sorted(label for label in seeds if label in process.filters_())
    producerSeeds = [label for label in graph.topologicalOrder(provenance)
                     if label in seeds and label not in filterSeeds and label not in wrapper]
    orderedSeeds = filterSeeds + producerSeeds

    droppedPaths = sorted(name for name in
                          list(process.paths_()) + list(process.endpaths_())
                          if name not in HLT_FINAL_PATHS and name not in requiredPaths)

    onPath = set(orderedSeeds) | wrapper | finalPathModules | keptPathModules
    setattr(process, taskName,
            cms.Task(*[getattr(process, label)
                       for label in sorted(set(provenance) - onPath)]))
    elements = prologue + [getattr(process, label) for label in orderedSeeds] + epilogue
    sequence = elements[0]
    for element in elements[1:]:
        sequence = sequence + element
    setattr(process, pathName, cms.Path(sequence, getattr(process, taskName)))
    process.setSchedule_(cms.Schedule(*([getattr(process, name) for name in keptPaths] +
                                        [getattr(process, pathName)] + finalPaths)))

    described = lambda label: {
        "label": label,
        "type": graph.modules[label].type_(),
        "edmType": "EDFilter" if label in process.filters_() else "EDProducer"}
    manifest = {
        "process": process.name_(),
        "moduleTags": list(moduleTags),
        "modifiers": _modifierNames(process),
        "path": pathName,
        "task": taskName,
        "schedule": [item.label_() for item in process.schedule_()],
        "requiredPaths": [{"path": name,
                           "consumedBy": requiredPaths[name][0],
                           "via": requiredPaths[name][1]} for name in keptPaths],
        "hltPrologue": [item.label_() for item in prologue],
        "hltEpilogue": [item.label_() for item in epilogue],
        "pathOrder": ([item.label_() for item in prologue] + orderedSeeds +
                      [item.label_() for item in epilogue]),
        "seeds": [described(label) for label in seeds],
        "dependencies": [dict(described(label),
                              consumedBy=provenance[label][0],
                              via=provenance[label][1])
                         for label in sorted(provenance)
                         if provenance[label] is not None and label in graph.modules],
        "boundary": [{"label": label,
                      "kind": entry["kind"],
                      "consumers": sorted({consumer for consumer, _ in entry["consumers"]})}
                     for label, entry in sorted(boundary.items())],
        "unscheduledTaggedModules": [label for label in graph.taggedModules()
                                     if label not in graph.scheduled],
        "droppedPaths": droppedPaths,
    }

    # the EventSetup closure has to be taken over everything that survives, which
    # includes the modules of the Paths that are kept whole
    keptModules = set(provenance) | finalPathModules | keptPathModules
    manifest["droppedEventSetup"] = pruneEventSetup(process, graph.framework,
                                                    keptModules, moduleTags)
    manifest["keptEventSetup"] = sorted(list(process.es_producers_()) +
                                        list(process.es_sources_()))

    # prune drops the Paths and EndPaths that are no longer in the Schedule, and
    # every producer, filter and analyzer that is not used by the ones that stay
    process.prune()
    return manifest
