#! /usr/bin/env python3
"""Check that DST_HeterogeneousReco lists the heterogeneous content of the menu.

The hand written DST_HeterogeneousReco path has to hold the heterogeneous modules
of the Phase-2 menu, together with everything needed to feed them.  Both sides of
that statement are computed here from the menu itself, so that the path cannot
fall behind a change to the menu without the difference being reported.

The menu is loaded on its own, with no era and no process modifiers, and every
heterogeneous module it defines is checked.  Which of them a given job runs
depends on the modifiers of the cmsDriver recipe, whereas the hand written path
has to cover the menu as a whole, so the check is deliberately independent of any
particular recipe.
"""

import collections
import importlib
import unittest

import FWCore.ParameterSet.Config as cms

from HLTrigger.Configuration.moduleClosure import (HLT_MENU,
                                                   buildFrameworkGraphFromProcess,
                                                   ModuleGraph,
                                                   parseTracerLog,
                                                   hltStructure,
                                                   moduleNamesOf)

MENU = "HLTrigger.Configuration.HLT_75e33_timing_cff"
HAND_WRITTEN_PATH = "DST_HeterogeneousReco"
HAND_WRITTEN_CFI = "%s.paths.%s_cfi" % (HLT_MENU, HAND_WRITTEN_PATH)


def menuProcess():
    """The menu on its own, with no era and no process modifiers."""
    process = cms.Process("HLT")
    process.load(MENU)
    return process


def heterogeneousContent(process):
    """{label: why it belongs} for the heterogeneous content of a menu.

    Determined the way the dumper determines it, from the tagged modules, their
    upstream closure and the fixed HLT structure they are wrapped in, except that
    every module the menu defines counts as available rather than only the ones
    the menu's own Schedule reaches.  That is what makes the answer the whole
    heterogeneous content of the menu instead of the content a particular set of
    process modifiers happens to turn on.

    Returns (content, provenance, graph, seeds) so that the report can group
    missing modules by the heterogeneous module that requires them and order
    them by data-flow dependency.
    """
    prologue, epilogue, _ = hltStructure(process)
    wrapper = set()
    for item in prologue + epilogue:
        wrapper |= moduleNamesOf(item)
    framework = parseTracerLog(buildFrameworkGraphFromProcess(process))
    graph = ModuleGraph(process, framework)
    graph.scheduled = set(graph.modules)
    seeds = graph.seeds()
    provenance, _, _ = graph.closure(seeds + sorted(wrapper - set(seeds)))

    content = dict((label, "part of the HLT structure") for label in wrapper)
    content.update((label, "heterogeneous module") for label in seeds)
    for label, origin in provenance.items():
        if origin is not None and label not in content:
            content[label] = "needed by %s" % origin[0]
    return content, provenance, graph, seeds


def handWrittenModules():
    """The modules of the hand written heterogeneous Path."""
    process = cms.Process("REF")
    process.load(HAND_WRITTEN_CFI)
    return set(getattr(process, HAND_WRITTEN_PATH).moduleNames())


def _traceToSeed(label, provenance, seeds):
    """Trace a label back through provenance to the seed that needs it.

    Returns the seed label, or the label itself if it is a seed or a wrapper
    module (provenance None) or if the chain cannot be followed.
    """
    seen = set()
    current = label
    while current not in seen:
        seen.add(current)
        origin = provenance.get(current)
        if origin is None:
            return current
        parent = origin[0]
        if parent in seeds:
            return parent
        current = parent
    return label


def report(missing, extra, content, provenance, graph, seeds):
    """A compact account of the difference, and of what to do about it."""
    lines = ["",
             "%s does not match the heterogeneous content of" % HAND_WRITTEN_PATH,
             "%s." % MENU,
             ""]
    if missing:
        # Group missing modules by the heterogeneous module that
        # requires them, and order each group by data-flow dependency.
        groups = collections.OrderedDict()
        for label in missing:
            root = _traceToSeed(label, provenance, seeds)
            groups.setdefault(root, []).append(label)
        ordered = graph.topologicalOrder(missing)
        rank = {label: i for i, label in enumerate(ordered)}
        lines.append("  missing from %s (%d):" % (HAND_WRITTEN_PATH, len(missing)))
        for root in sorted(groups):
            group = sorted(groups[root], key=lambda l: rank.get(l, 0))
            lines.append("    Modules needed by heterogeneous module %s:" % root)
            for label in group:
                lines.append("      %s  # %s" % (label, content[label]))
        lines.append("")
    if extra:
        lines.append("  in %s but unused by the menu (%d):"
                     % (HAND_WRITTEN_PATH, len(extra)))
        lines += ["    %s" % label for label in extra]
        lines.append("")
    lines += ["  To fix, add or remove those modules in",
              "    %s" % importlib.import_module(HAND_WRITTEN_CFI).__file__,
              "",
              "  To see the whole configuration the dumper produces, including the",
              "  EventSetup and the L1 emulation left out of this comparison, dump",
              "  a cmsDriver configuration built on this menu:",
              "    hltDumpTaggedModules --manifest manifest.json -o het.py <config>.py",
              "  The manifest records which dependency was pulled in by what.",
              ""]
    return "\n".join(lines)


class TestHeterogeneousMenu(unittest.TestCase):
    def testHandWrittenPathMatchesTheMenu(self):
        content, provenance, graph, seeds = heterogeneousContent(menuProcess())
        found = handWrittenModules()
        missing = sorted(set(content) - found)
        extra = sorted(found - set(content))
        if missing or extra:
            self.fail(report(missing, extra, content, provenance, graph, seeds))


if __name__ == "__main__":
    unittest.main(verbosity=2)
