# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

"""Apply one truth-graph selection preset to every module that has to agree on it.

The preset decides which particle is the signal, so several modules have to read the
same answer: the graph producer builds the selected view, the targets producer publishes
the signal-seed denominators, and every validator books its signal folders from the same
seeds. Setting them separately lets them drift, and a drifted signal denominator is not
visible in any plot.

Pick the preset by name, or name the generator fragment and let the rules in
``truthGraphSelections`` resolve it:

    from PhysicsTools.TruthInfo.customiseTruthPreset import applyTruthPreset
    applyTruthPreset(process, preset="top")
    applyTruthPreset(process, fragment="TTbar_14TeV_TuneCP5_cfi")
    applyTruthPreset(process, preset="top", seedParentDepth=2)

From cmsDriver, either as a command:

    --customise_commands "from PhysicsTools.TruthInfo.customiseTruthPreset import \\
        applyTruthPreset; applyTruthPreset(process, preset='top')"

or as a plain customise reading the environment:

    TRUTH_GRAPH_PRESET=top cmsDriver.py ... \\
        --customise PhysicsTools/TruthInfo/customiseTruthPreset.customiseTruthPreset

Every other field keeps the value the chain gave it, so a preset does not undo the
pile-up wiring or the reconstructable species of a mixed job.
"""

import os

import FWCore.ParameterSet.Config as cms

from PhysicsTools.TruthInfo.truthGraphSelections import (
    TEMPLATE_NAMES,
    selectionForFragment,
    templateForFragment,
)

# The producer that builds the selected view. Every module that carries seed parameters
# is found by name below, so a new validator needs no edit here.
GRAPH_PRODUCER = "truthLogicalGraphProducer"
SEED_PARAMETERS = ("signalSeedPdgIds", "signalSeedHadronFlavors")

# The postProcessing fields the preset owns. Anything outside this list, such as
# reconstructablePdgIds or dropHitlessSimSubgraphs, is left as the chain set it unless
# the caller overrides it by name.
_PRESET_FIELDS = (
    "seedPdgIds",
    "seedHadronFlavors",
    "seedParentDepth",
    "decayPdgIdGroups",
    "keepStableSpectators",
    "attachSelectionSources",
    "keepProductionSiblings",
    "signalOnly",
    "keepBunchCrossings",
)

_TYPES = {
    "seedPdgIds": lambda v: cms.vint32(*v),
    "seedHadronFlavors": lambda v: cms.vint32(*v),
    "seedParentDepth": lambda v: cms.uint32(v),
    "decayPdgIdGroups": lambda v: cms.VPSet(*[cms.PSet(pdgIds=cms.vint32(*g)) for g in v]),
    "keepStableSpectators": lambda v: cms.bool(v),
    "attachSelectionSources": lambda v: cms.bool(v),
    "keepProductionSiblings": lambda v: cms.bool(v),
    "signalOnly": lambda v: cms.bool(v),
    "keepBunchCrossings": lambda v: cms.vint32(*v),
    "collapseIntermediateGenParticles": lambda v: cms.bool(v),
    "reconstructablePdgIds": lambda v: cms.vint32(*v),
    "dropHitlessSimSubgraphs": lambda v: cms.bool(v),
    "ignoredPdgIds": lambda v: cms.vint32(*v),
    "ignoredParticleIds": lambda v: cms.vuint32(*v),
}


def resolvePreset(preset=None, fragment=None):
    """The preset name for an explicit choice or for a generator fragment."""
    if preset is not None and fragment is not None:
        raise ValueError("applyTruthPreset takes a preset or a fragment, not both")
    if preset is not None:
        if preset not in TEMPLATE_NAMES:
            raise KeyError("unknown preset %r (known: %s)" % (preset, ", ".join(TEMPLATE_NAMES)))
        return preset
    if fragment is None:
        raise ValueError("applyTruthPreset needs a preset or a fragment")
    return templateForFragment(fragment)[0]


def applyTruthPreset(process, preset=None, fragment=None, **overrides):
    """Set the selection on the graph producer and the matching seeds on every module that
    reads them. Returns the process, so it chains like any other customise."""
    name = resolvePreset(preset=preset, fragment=fragment)
    selection = selectionForFragment(name=fragment, template=preset, **overrides)

    fields = dict((field, selection[field]) for field in _PRESET_FIELDS)
    for field, value in overrides.items():
        if field not in _TYPES:
            raise KeyError("unknown postProcessing field %r" % field)
        fields[field] = value

    if hasattr(process, GRAPH_PRODUCER):
        postProcessing = getattr(process, GRAPH_PRODUCER).postProcessing
        for field, value in fields.items():
            setattr(postProcessing, field, _TYPES[field](value))

    # The signal-seed denominator is the preset's own signal object, so every module that
    # reads seeds reads the same ones: the targets producer publishes the denominators and
    # each validator books its signal folders from them. [0] is the full-graph escape
    # hatch, not a species.
    seeds = [p for p in selection["seedPdgIds"] if p != 0]
    flavors = list(selection["seedHadronFlavors"])
    seeded = []
    modules = {}
    # A DQM harvester is an EDProducer, so producers_() has to be read as well.
    for kind in ("producers_", "analyzers_", "filters_"):
        modules.update(getattr(process, kind)())
    for label, module in modules.items():
        if not all(hasattr(module, name) for name in SEED_PARAMETERS):
            continue
        module.signalSeedPdgIds = cms.vint32(*seeds)
        module.signalSeedHadronFlavors = cms.vint32(*flavors)
        seeded.append(label)

    print("[truth] selection preset '%s'%s: seedPdgIds=%s seedHadronFlavors=%s seedParentDepth=%d, "
          "seeds set on %d module(s): %s" %
          (name,
           " from fragment '%s'" % fragment if fragment else "",
           selection["seedPdgIds"],
           selection["seedHadronFlavors"],
           selection["seedParentDepth"],
           len(seeded), ", ".join(sorted(seeded)) or "none"))
    return process


def customiseTruthPreset(process):
    """Zero-argument customise for cmsDriver. Reads TRUTH_GRAPH_PRESET, or
    TRUTH_GRAPH_FRAGMENT for a generator fragment. With neither set it does nothing,
    which leaves the whole graph."""
    preset = os.environ.get("TRUTH_GRAPH_PRESET")
    fragment = os.environ.get("TRUTH_GRAPH_FRAGMENT")
    if preset is None and fragment is None:
        print("[truth] no TRUTH_GRAPH_PRESET or TRUTH_GRAPH_FRAGMENT set, keeping the whole graph")
        return process
    return applyTruthPreset(process, preset=preset, fragment=fragment)
