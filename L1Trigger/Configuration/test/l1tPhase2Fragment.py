#!/usr/bin/env python3
"""Generate and validate the standalone Phase-2 L1 configuration (L1_Phase2_cff + L1_Phase2/,
and the same content in a single file, L1_Phase2_expanded_cff), and the complete L1 process
test/OnLine_L1_Phase2.py.

The L1 content is the difference between two cmsDriver configurations of the reference
workflow: one running only the base step (DIGI) and one running base + L1 steps. Only
objects reachable from the L1 paths are kept; the Run-3 legacy chain (everything feeding
only simGtStage2Digis) is moved to its own task and path.

  l1tPhase2Fragment.py generate [--outdir L1Trigger/Configuration/python]
  l1tPhase2Fragment.py validate

See L1Trigger/Configuration/python/L1_Phase2/README.md and https://github.com/artlbv/cmssw/issues/10
"""
import argparse
import collections
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.Types import _OptionalParameter

# reference workflow: 38434.0 (TTbar_14TeV, Run4D128, the Run4 default), step2
DRIVER_OPTIONS = [
    "--conditions", "auto:phase2_realistic_T35",
    "--geometry", "ExtendedRun4D128",
    "--era", "Phase2C26I13M9",
    "--datatier", "GEN-SIM-DIGI-RAW",
    "--eventcontent", "FEVTDEBUGHLT",
    "--filein", "file:step1.root",
    "--fileout", "file:step2.root",
    "-n", "10",
    "--mc",
    "--no_exec",
]
BASE_STEPS = "DIGI:pdigi_valid"
L1_STEPS = "L1TrackTrigger,L1,L1P2GT"

# modules whose ancestors (unless also needed by Phase-2 objects) form the Run-3 legacy chain
LEGACY_ROOTS = ["simGtStage2Digis", "simBmtfDigis", "simCscTriggerPrimitiveDigisRun3"]
LEGACY_TASK = "L1TLegacyRun3Task"
LEGACY_PATH = "L1simulationLegacy_step"
L1_PATH_AFTER_WHICH_LEGACY = "L1simulation_step"

# MC-truth modules: the TT*Associator* plugins and everything reading their products. They run in their own
# task and path, which customisePhase2TTNoMC.customiseL1P2NoMCTruth drops to run the L1 emulation on stored
# events (the tracker sim links are not stored). Nothing else may depend on them.
TRUTH_TYPES = re.compile(r"^TT\w*Associator")
TRUTH_TASK = "L1TrackTriggerMCTruthTask"
TRUTH_PATH = "L1TrackTriggerMCTruth_step"
TRUTH_CUSTOMISE = "L1Trigger/Configuration/customisePhase2TTNoMC.customiseL1P2NoMCTruth"
# inputs a plugin only reads when a flag is set: {plugin: (flag, [InputTag parameters])}
CONDITIONAL_INPUTS = {
    "L1FPGATrackProducer": ("readMoreMcTruth", ["MCTruthClusterInputTag", "MCTruthStubInputTag", "TrackingParticleInputTag"]),
}
SPLIT_PATHS = (LEGACY_PATH, TRUTH_PATH)  # not in the cmsDriver process

FRAGMENT = "L1_Phase2"
EXPANDED = FRAGMENT + "_expanded"

# complete process running only the L1 emulation (cmsDriver step L1P2) on the GEN-SIM-DIGI-RAW output
# of the reference workflow, analogous to HLTrigger/Configuration/test/OnLine_HLT_GRun.py
ONLINE = "OnLine_" + FRAGMENT
ONLINE_PROCESS = "L1P2"
ONLINE_FILES = {"file:step1.root": "file:step2.root", "file:step2.root": f"file:{ONLINE}.root"}
ONLINE_OPTIONS = [ONLINE_FILES.get(o, o) for o in DRIVER_OPTIONS] + ["--processName", ONLINE_PROCESS,
                                                                       "--customise", TRUTH_CUSTOMISE]
ONLINE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ONLINE + ".py")
PACKAGE = "L1Trigger/Configuration"
PATH_LIST = "l1Paths"

MODULE_TYPES = (cms.EDProducer, cms.EDFilter, cms.EDAnalyzer)
SEQUENCE_TYPES = (cms.Sequence, cms.Task)


def runDriver(steps, cfgFile):
    cmd = ["cmsDriver.py", "step2", "-s", steps, "--python_filename", cfgFile] + DRIVER_OPTIONS
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def loadProcess(cfgFile):
    spec = importlib.util.spec_from_file_location(os.path.basename(cfgFile)[:-3], cfgFile)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.process


def driverProcesses(workdir):
    base, full = os.path.join(workdir, "base_cfg.py"), os.path.join(workdir, "full_cfg.py")
    runDriver(BASE_STEPS, base)
    runDriver(BASE_STEPS + "," + L1_STEPS, full)
    processes = loadProcess(base), loadProcess(full)
    for process in processes:
        for objects in (process.producers_(), process.filters_(), process.analyzers_(), process.es_sources_(),
                        process.es_producers_(), process.psets_()):
            for obj in objects.values():
                dropPlaceholders(obj)
    return processes


def dropPlaceholders(pset):
    """Removes the placeholders of the fillDescriptions-generated cfis, which set nothing: unset optional
    parameters (cms.optional.X) and VPSet templates (cms.PSetTemplate). The configuration database does not
    store them."""
    for name in pset.parameterNames_():
        value = getattr(pset, name)
        if isinstance(value, _OptionalParameter) and value.value() is None:
            delattr(pset, name)
        elif isinstance(value, cms.PSet):
            dropPlaceholders(value)
        elif isinstance(value, cms.VPSet):
            value._template = None
            for item in value:
                dropPlaceholders(item)


class Collector:
    """Collects module and sequence/task labels reachable from the visited paths."""

    def __init__(self):
        self.modules = set()
        self.sequences = set()

    def enter(self, node):
        if isinstance(node, MODULE_TYPES):
            self.modules.add(node.label_())
        elif isinstance(node, SEQUENCE_TYPES) and node.hasLabel_():
            self.sequences.add(node.label_())

    def leave(self, node):
        pass


def reachable(paths):
    collector = Collector()
    for path in paths:
        path.visit(collector)
    return collector


def inputTags(pset):
    for name in pset.parameterNames_():
        par = getattr(pset, name)
        if isinstance(par, cms.InputTag):
            yield par
        elif isinstance(par, cms.VInputTag):
            yield from (t if isinstance(t, cms.InputTag) else cms.InputTag(t) for t in par)
        elif isinstance(par, cms.PSet):
            yield from inputTags(par)
        elif isinstance(par, cms.VPSet):
            for p in par:
                yield from inputTags(p)


def psetRefs(pset):
    """Labels of the top-level PSets referenced (refToPSet_) from 'pset'."""
    for name in pset.parameterNames_():
        par = getattr(pset, name)
        for p in (par if isinstance(par, cms.VPSet) else [par] if isinstance(par, cms.PSet) else []):
            if p.isRef_():
                yield p.refToPSet_.value()
            else:
                yield from psetRefs(p)


def referencedPSets(process, objects):
    """Top-level PSets referenced from 'objects', following references between top-level PSets."""
    stack, seen = [r for obj in objects for r in psetRefs(obj)], {}
    while stack:
        label = stack.pop()
        if label not in seen:
            seen[label] = process.psets_()[label]
            stack.extend(psetRefs(seen[label]))
    return seen


def moduleInputs(module):
    """InputTags a module reads, without those it only reads when a flag (CONDITIONAL_INPUTS) is set."""
    flag, params = CONDITIONAL_INPUTS.get(module.type_(), (None, []))
    if flag is not None and getattr(module, flag).value():
        params = []
    ignored = {getattr(module, p).value() for p in params if hasattr(module, p)}
    for tag in inputTags(module):
        if tag.value() not in ignored:
            yield tag


def descendants(roots, deps):
    users = collections.defaultdict(set)
    for module, inputs in deps.items():
        for i in inputs:
            users[i].add(module)
    return ancestors(roots, users)


def ancestors(roots, deps):
    seen, stack = set(), list(roots)
    while stack:
        label = stack.pop()
        if label not in seen:
            seen.add(label)
            stack.extend(deps.get(label, ()))
    return seen


Content = collections.namedtuple("Content", "paths modules sequences es psets legacy truth externalInputs")


def l1Content(base, full):
    """Returns the L1 paths, modules, sequences, ES objects, top-level PSets, legacy and MC-truth modules and
    external inputs."""
    basePaths = {p.label_() for p in base.schedule}
    paths = [p for p in full.schedule if p.label_() not in basePaths]
    reached = reachable(paths)
    modules = reached.modules
    deps = {m: {t.getModuleLabel() for t in moduleInputs(getattr(full, m))} & modules for m in modules}
    legacyChain = ancestors([r for r in LEGACY_ROOTS if r in modules], deps)
    legacy = legacyChain - ancestors(modules - legacyChain, deps)
    truth = descendants([m for m in modules if TRUTH_TYPES.match(getattr(full, m).type_())], deps)
    needTruth = sorted(m for m in modules - truth if deps[m] & truth)
    if needTruth:
        raise RuntimeError(f"modules outside the MC-truth chain read MC-truth products: {needTruth}")
    externalInputs = sorted({t.getModuleLabel() for m in modules for t in moduleInputs(getattr(full, m))
                             if t.getModuleLabel() and t.getModuleLabel() not in modules})
    es = {}
    for kind in ("es_sources_", "es_producers_", "es_prefers_"):
        objs, baseObjs = getattr(full, kind)(), getattr(base, kind)()
        for label in objs:
            if label not in baseObjs:
                es[label] = objs[label]
            elif objs[label].dumpPython() != baseObjs[label].dumpPython():
                raise RuntimeError(f"ES object '{label}' is modified by the L1 steps")
    psets = referencedPSets(full, [getattr(full, m) for m in modules] + list(es.values()))
    for label, pset in psets.items():
        if label in base.psets_() and base.psets_()[label].dumpPython() != pset.dumpPython():
            raise RuntimeError(f"top-level PSet '{label}' is modified by the L1 steps")
    return Content(paths, modules, reached.sequences, es, psets, legacy, truth, externalInputs)


def removeDirectly(container, item):
    """Removes 'item' only where 'container' holds it directly: remove() would replace nested
    tasks/sequences holding it by unlabelled copies."""
    if isinstance(container, cms.Task):
        if item in container._collection:
            container._collection.remove(item)
            return True
        return False
    return container._replaceIfHeldDirectly(item, None)


def splitOff(full, paths, sequences, moved, taskLabel, pathLabel, after):
    """Removes the 'moved' modules from the L1 sequences/tasks and schedules them in their own task and
    path, inserted after the path 'after'."""
    containers = [getattr(full, s) for s in sorted(sequences)]
    for label in sorted(moved):
        module = getattr(full, label)
        for container in containers:
            removeDirectly(container, module)
    setattr(full, taskLabel, cms.Task(*[getattr(full, m) for m in sorted(moved)]))
    setattr(full, pathLabel, cms.Path(getattr(full, taskLabel)))
    index = [p.label_() for p in paths].index(after) + 1
    paths.insert(index, getattr(full, pathLabel))
    # drop the containers that became empty
    empty = sorted(s for s in reachable(paths).sequences if not reachable([getattr(full, s)]).modules)
    for label in empty:
        for container in containers + paths:
            removeDirectly(container, getattr(full, label))
    return reachable(paths).sequences


HEADER = "import FWCore.ParameterSet.Config as cms\n\n"


def dumpContainer(obj, kinds):
    """Dump of a path/sequence/task with 'process.' references turned into imports."""
    text = obj.dumpPython()
    names = sorted(set(re.findall(r"process\.(\w+)", text)))
    imports = "".join(f"from ..{kinds[n]}.{n}_cfi import *\n" for n in names)
    return imports, re.sub(r"process\.", "", text)


def writeFile(directory, label, imports, body):
    with open(os.path.join(directory, label + "_cfi.py"), "w") as out:
        out.write(HEADER + imports + ("\n" if imports else "") + f"{label} = {body}")


def expandedSequenceOrder(full, sequences):
    """Sequence/task labels ordered so that each one is defined after those it references."""
    refs = {s: set(re.findall(r"process\.(\w+)", getattr(full, s).dumpPython())) & sequences for s in sequences}
    order, done = [], set()
    while len(order) < len(sequences):
        ready = sorted(s for s in sequences - done if refs[s] <= done)
        if not ready:
            raise RuntimeError("circular sequence references: " + ", ".join(sorted(sequences - done)))
        order += ready
        done.update(ready)
    return order


def fragmentHeader(externalInputs, layout):
    return [HEADER.rstrip("\n"), "",
            "# Standalone, fully resolved configuration of the Phase-2 L1 trigger emulation",
            "# (track trigger, trigger primitives, correlator/GTT/GMT/calo emulators, P2GT menu),",
            "# modelled on HLTrigger/Configuration/python/HLT_75e33_cff.py.",
            "# " + layout,
            "#",
            "# GENERATED by L1Trigger/Configuration/test/l1tPhase2Fragment.py: do not edit by hand.",
            f"#   cmsDriver.py step2 -s {BASE_STEPS},{L1_STEPS} {' '.join(DRIVER_OPTIONS)}",
            f"#   release: {os.environ.get('CMSSW_VERSION', 'unknown')}",
            "# Geometry, magnetic field and GlobalTag are not included (taken from the driver).",
            "# Required inputs (not produced here): " + ", ".join(externalInputs),
            "",
            f'fragment = cms.ProcessFragment("L1")',
            ""]


def pathList(paths):
    return (["# Paths in schedule order. There is no fragment.schedule, so loading the fragment does not",
             "# replace process.schedule: the cmsDriver step L1P2 (or the user) appends these paths."]
            + [f"{PATH_LIST} = ["] + [f'    "{p.label_()}",' for p in paths] + ["]", ""])


def writeExpanded(outdir, full, paths, modules, sequences, es, psets, externalInputs):
    """Same content as L1_Phase2_cff + L1_Phase2/, with every object defined in one file."""
    def define(label, obj):
        return f"fragment.{label} = " + re.sub(r"process\.", "fragment.", obj.dumpPython()).rstrip("\n")

    lines = fragmentHeader(externalInputs, f"Single-file version of {FRAGMENT}_cff.py + {FRAGMENT}/ (same content).")
    lines += ["### Event-setup (ES sources, ES producers, ES prefers)", ""]
    lines += [define(label, es[label]) + "\n" for label in sorted(es)]
    if psets:
        lines += ["### Top-level PSets (referenced by modules with refToPSet_)", ""]
        lines += [define(label, psets[label]) + "\n" for label in sorted(psets)]
    lines += ["### Modules", ""]
    lines += [define(label, getattr(full, label)) + "\n" for label in sorted(modules)]
    lines += ["### Sequences and tasks", ""]
    lines += [define(label, getattr(full, label)) + "\n" for label in expandedSequenceOrder(full, sequences)]
    lines += ["### Paths", ""]
    lines += [define(p.label_(), p) for p in paths] + [""]
    lines += pathList(paths)
    with open(os.path.join(outdir, EXPANDED + "_cff.py"), "w") as out:
        out.write("\n".join(lines))


def onlineCommand(cfgFile):
    return ["cmsDriver.py", "l1", "-s", "L1P2", "--python_filename", cfgFile] + ONLINE_OPTIONS


def writeOnline(outfile):
    """Complete process of the cmsDriver step L1P2 (which loads L1_Phase2_cff), fully expanded."""
    with tempfile.TemporaryDirectory() as workdir:
        cfgFile = os.path.join(workdir, "online_cfg.py")
        subprocess.run(onlineCommand(cfgFile), check=True, stdout=subprocess.DEVNULL, cwd=workdir)
        dump = subprocess.run(["edmConfigDump", cfgFile], check=True, capture_output=True, text=True, cwd=workdir).stdout
    header = ["# Complete process running the standalone Phase-2 L1 emulation (L1_Phase2_cff, cmsDriver step L1P2)",
              "# on the GEN-SIM-DIGI-RAW output of the reference workflow, analogous to",
              "# HLTrigger/Configuration/test/OnLine_HLT_GRun.py:  cmsRun " + ONLINE + ".py",
              "#",
              "# GENERATED by L1Trigger/Configuration/test/l1tPhase2Fragment.py: do not edit by hand.",
              "#   " + " ".join(onlineCommand(ONLINE + ".py")),
              f"#   release: {os.environ.get('CMSSW_VERSION', 'unknown')}",
              "# GlobalTag, geometry, source and output come from the cmsDriver options above.",
              "", ""]
    with open(outfile, "w") as out:
        out.write("\n".join(header) + dump)


def generate(args):
    with tempfile.TemporaryDirectory() as workdir:
        base, full = driverProcesses(workdir)
    paths, modules, sequences, es, psets, legacy, truth, externalInputs = l1Content(base, full)
    sequences = splitOff(full, paths, sequences, legacy, LEGACY_TASK, LEGACY_PATH, L1_PATH_AFTER_WHICH_LEGACY)
    sequences = splitOff(full, paths, sequences, truth, TRUTH_TASK, TRUTH_PATH, LEGACY_PATH)

    top = os.path.join(args.outdir, FRAGMENT)
    for kind in ("modules", "eventsetup", "sequences", "paths"):
        directory = os.path.join(top, kind)
        shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory)
    kinds = {m: "modules" for m in modules}
    kinds.update({s: "sequences" for s in sequences})
    kinds.update({p.label_(): "paths" for p in paths})

    for label in sorted(modules):
        writeFile(os.path.join(top, "modules"), label, "", getattr(full, label).dumpPython())
    for label in sorted(es):
        writeFile(os.path.join(top, "eventsetup"), label, "", es[label].dumpPython())
    os.makedirs(os.path.join(top, "psets"), exist_ok=True)
    for label in sorted(psets):
        writeFile(os.path.join(top, "psets"), label, "", psets[label].dumpPython())
    for label in sorted(sequences):
        writeFile(os.path.join(top, "sequences"), label, *dumpContainer(getattr(full, label), kinds))
    for path in paths:
        writeFile(os.path.join(top, "paths"), path.label_(), *dumpContainer(path, kinds))

    loadPrefix = f"{PACKAGE}/{FRAGMENT}"
    lines = fragmentHeader(externalInputs, f"One file per object in {FRAGMENT}/; the same content in a single file: {EXPANDED}_cff.py.")
    lines += ["### Event-setup (ES sources, ES producers, ES prefers)"]
    lines += [f'fragment.load("{loadPrefix}/eventsetup/{label}_cfi")' for label in sorted(es)]
    if psets:
        lines += ["", "### Top-level PSets (referenced by modules with refToPSet_)"]
        lines += [f'fragment.load("{loadPrefix}/psets/{label}_cfi")' for label in sorted(psets)]
    lines += ["", "### Paths (modules and sequences are imported by the path files)"]
    lines += [f'fragment.load("{loadPrefix}/paths/{p.label_()}_cfi")' for p in paths]
    lines += [""] + pathList(paths)
    with open(os.path.join(args.outdir, FRAGMENT + "_cff.py"), "w") as out:
        out.write("\n".join(lines))
    writeExpanded(args.outdir, full, paths, modules, sequences, es, psets, externalInputs)
    writeOnline(args.online)

    print(f"wrote {len(modules)} modules ({len(legacy)} Run-3 legacy, {len(truth)} MC truth: {', '.join(sorted(truth))}), "
          f"{len(es)} ES objects, {len(psets)} PSets, "
          f"{len(sequences)} sequences/tasks, {len(paths)} paths to {top} and {EXPANDED}_cff.py; "
          f"complete process: {args.online}")
    print("required inputs:", ", ".join(externalInputs))


def validate(args):
    """Compares base + L1_Phase2_cff (and L1_Phase2_expanded_cff) with the cmsDriver process:
    modules, parameters, ES, schedule."""
    with tempfile.TemporaryDirectory() as workdir:
        base, full = driverProcesses(workdir)
        tests = {name: loadProcess(os.path.join(workdir, "base_cfg.py")) for name in (FRAGMENT, EXPANDED)}
    ref = l1Content(base, full)
    results = [validateFragment(name, test, full, ref) for name, test in tests.items()]
    results.append(validateOnline(full, ref))
    return max(results)


def report(name, errors, ref, exclude=()):
    for e in errors:
        print("ERROR:", e)
    print(f"{name}: checked {len(ref.modules - set(exclude))} modules, {len(ref.es)} ES objects, {len(ref.psets)} PSets, "
          f"{len(ref.paths)} paths: " + ("FAILED" if errors else "OK"))
    return 1 if errors else 0


def validateFragment(name, test, full, ref):
    baseSchedule = [p.label_() for p in test.schedule]
    test.load(f"{PACKAGE.replace('/', '.')}.{name}_cff")
    fragmentPaths = getattr(importlib.import_module(f"{PACKAGE.replace('/', '.')}.{name}_cff"), PATH_LIST)
    errors = []
    if [p.label_() for p in test.schedule] != baseSchedule:
        errors.append("loading the fragment changes process.schedule")
    errors += compareContent(test, fragmentPaths, full, ref)
    return report(name + "_cff", errors, ref)


def validateOnline(full, ref):
    """The L1 content of test/OnLine_L1_Phase2.py, without the MC-truth path, must match the cmsDriver
    process (GlobalTag, geometry and I/O are not compared, so that an autoCond update alone does not fail
    the check)."""
    if not os.path.exists(ONLINE_FILE):
        return report(ONLINE + ".py", [f"{ONLINE_FILE} not found"], ref, ref.truth)
    test = loadProcess(ONLINE_FILE)
    fragmentPaths = getattr(importlib.import_module(f"{PACKAGE.replace('/', '.')}.{FRAGMENT}_cff"), PATH_LIST)
    l1Paths = [p for p in fragmentPaths if p != TRUTH_PATH]
    scheduled = [p.label_() for p in test.schedule]
    errors = []
    if [p for p in scheduled if p in fragmentPaths] != l1Paths:
        errors.append(f"the paths of {PATH_LIST} except {TRUTH_PATH} must be scheduled, in order")
    errors += compareContent(test, l1Paths, full, ref, exclude=ref.truth)
    return report(ONLINE + ".py", errors, ref, ref.truth)


def compareContent(test, pathLabels, full, ref, exclude=()):
    """Compares modules (except 'exclude'), ES objects, top-level PSets and path order of 'test' with the
    cmsDriver process."""
    refPaths, refES, refPSets = ref.paths, ref.es, ref.psets
    refModules = ref.modules - set(exclude)
    testPaths = [getattr(test, p) for p in pathLabels if hasattr(test, p)]
    testModules = reachable(testPaths).modules
    errors = []
    if len(testPaths) != len(pathLabels):
        errors.append(f"paths listed in {PATH_LIST} but not defined: {sorted(set(pathLabels) - set(test.paths_()))}")

    missing, extra = refModules - testModules, testModules - refModules
    if missing:
        errors.append(f"modules missing from the fragment: {sorted(missing)}")
    if extra:
        errors.append(f"modules not in the cmsDriver process: {sorted(extra)}")
    for label in sorted(refModules & testModules):
        if getattr(full, label).dumpPython() != getattr(test, label).dumpPython():
            errors.append(f"parameters differ for module '{label}'")

    testES = {}
    for kind in ("es_sources_", "es_producers_", "es_prefers_"):
        testES.update(getattr(test, kind)())
    # ES prefers are matched by content: their labels are not kept by a dump (edmConfigDump renames them)
    testPrefers = {p.dumpPython() for p in test.es_prefers_().values()}
    for label, obj in refES.items():
        if isinstance(obj, cms.ESPrefer):
            if obj.dumpPython() not in testPrefers:
                errors.append(f"ES prefer missing from the fragment: '{label}'")
        elif label not in testES:
            errors.append(f"ES object missing from the fragment: '{label}'")
        elif obj.dumpPython() != testES[label].dumpPython():
            errors.append(f"parameters differ for ES object '{label}'")

    for label, pset in refPSets.items():
        if label not in test.psets_():
            errors.append(f"top-level PSet missing from the fragment: '{label}'")
        elif pset.dumpPython() != test.psets_()[label].dumpPython():
            errors.append(f"parameters differ for top-level PSet '{label}'")

    refOrder = [p.label_() for p in refPaths]
    testOrder = [p for p in pathLabels if p not in SPLIT_PATHS]
    if testOrder != refOrder:
        errors.append(f"{PATH_LIST} differs from the cmsDriver path order")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="regenerate L1_Phase2_cff.py, L1_Phase2/, L1_Phase2_expanded_cff.py and test/OnLine_L1_Phase2.py")
    gen.add_argument("--outdir", default=os.path.join(os.environ.get("CMSSW_BASE", "."), "src", PACKAGE, "python"))
    gen.add_argument("--online", default=ONLINE_FILE, help="complete L1 process file (default: %(default)s)")
    sub.add_parser("validate", help="compare both fragments and the complete L1 process with the cmsDriver configuration")
    args = parser.parse_args()
    return generate(args) if args.command == "generate" else validate(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
