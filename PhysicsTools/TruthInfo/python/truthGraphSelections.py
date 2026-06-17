# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# Part of the MC-truth-graph prototype - under heavy development, not yet open
# to external contributions (see PhysicsTools/TruthInfo/README.md).

"""Per-process logical-graph selection presets for the enableTruth relval samples.

The enableTruth process modifier attaches to *every* Run4 workflow, i.e. to ~140
generator fragments. They collapse to seven archetypes that fix the right
``postProcessing`` selection (which particle is the seed, whether to pull in the
seed's hard-scatter co-products, which decay channel to keep, ...):

  gun          single/multi-particle guns          seed = the gun species
  resonance    s-channel Z / Drell-Yan / Z' / W     seed = the resonance, ISR context
  vbf          VBF / t-channel Higgs               seed = Higgs + keepProductionSiblings
  ggf          ggF / s-channel single Higgs        seed = Higgs
  top          ttbar / single top / t'              seed = top
  heavyflavor  B / charmonium / bottomonium         seed by heavy-flavor content
  full         QCD / MinBias / NuGun / unknown      keep the whole graph (no selection)

``selectionForFragment(name)`` maps a generator-fragment (or short gallery label)
to one of these and returns a plain dict of ``postProcessing`` parameters; the
preset is only a starting point - pass keyword overrides to customise any field,
or build a config by hand from ``TEMPLATES``. ``postProcessingPSet`` wraps the
result as a ``cms.PSet`` for use in a producer/dumper config, and the module's
CLI prints the matching ``dumpTruthGraphsFromGENSIMRECO_cfg.py`` flags (used by
makeTruthGallery.sh).
"""

import re

# The seven archetype names, in resolution priority is handled by RULES below.
TEMPLATE_NAMES = ("gun", "resonance", "vbf", "ggf", "top", "heavyflavor", "full")


def _selection(seedPdgIds=(0,),
               seedHadronFlavors=(),
               seedParentDepth=0,
               decayPdgIdGroups=(),
               keepStableSpectators=True,
               attachSelectionSources=True,
               keepProductionSiblings=False):
    """A full postProcessing selection dict with sensible build-side defaults."""
    return dict(
        seedPdgIds=[int(p) for p in seedPdgIds],
        seedHadronFlavors=[int(f) for f in seedHadronFlavors],
        seedParentDepth=int(seedParentDepth),
        decayPdgIdGroups=[[int(p) for p in g] for g in decayPdgIdGroups],
        keepStableSpectators=bool(keepStableSpectators),
        attachSelectionSources=bool(attachSelectionSources),
        keepProductionSiblings=bool(keepProductionSiblings),
    )


# --- the seven pre-made templates ------------------------------------------
# Each is a zero-argument factory returning a fresh dict (so callers can mutate).
TEMPLATES = {
    # Guns: each primary is its own signal; no upstream, no underlying event.
    "gun": lambda: _selection(seedPdgIds=(0,), seedParentDepth=0,
                              keepStableSpectators=False, attachSelectionSources=True),
    # s-channel resonance: seed the boson, show one generation of incoming partons.
    "resonance": lambda: _selection(seedPdgIds=(23, 32), seedParentDepth=1),
    # VBF / t-channel: the tagging quarks recoil at the Higgs production vertex.
    "vbf": lambda: _selection(seedPdgIds=(25,), seedParentDepth=1, keepProductionSiblings=True),
    # ggF / s-channel single Higgs: 2->1, no production-vertex co-products.
    "ggf": lambda: _selection(seedPdgIds=(25,), seedParentDepth=1),
    # Top: seed both tops; their decay chains are the signal.
    "top": lambda: _selection(seedPdgIds=(6, -6), seedParentDepth=1),
    # Heavy flavor: seed by hadron flavor content (5=b, 4=c); the hadron is the root.
    "heavyflavor": lambda: _selection(seedPdgIds=(), seedHadronFlavors=(5,), seedParentDepth=0,
                                      keepStableSpectators=False),
    # Everything else (QCD, MinBias, NuGun, BSM, unknown): keep the whole graph.
    "full": lambda: _selection(seedPdgIds=(0,)),
}


# --- gun species -> seed PDG ids -------------------------------------------
# The species word is glued to the multiplicity prefix (SingleElectron, TenTau,
# FourMu), so this is a plain ordered substring search: specific/long species
# first, ambiguous short ones (mu, pi) last. Detector-region CloseBy / CE_ guns
# have a configurable species and fall through to the full-graph seed (0).
_GUN_SPECIES = (
    ("electron", [11, -11]),
    ("positron", [-11]),
    ("gamma", [22]),
    ("photon", [22]),
    ("muon", [13, -13]),
    ("pion", [211, -211]),
    ("proton", [2212]),
    ("tau", [15, -15]),
    ("kaon", [321, -321]),
    ("nu", [0]),          # neutrino guns leave no hits -> keep the full graph
    ("mu", [13, -13]),
    ("pi", [211, -211]),
)


def gunSeed(name):
    """Seed PDG ids for a particle-gun fragment, or [0] (full graph) if unknown."""
    low = name.lower()
    for token, pdgs in _GUN_SPECIES:
        if token in low:
            return list(pdgs)
    return [0]


# --- fragment-name -> (template, overrides) rules --------------------------
# First match wins; order specific -> generic. Patterns match both the full
# generator-fragment cfi name and the short gallery label.
_RULES = (
    (r"(?i)(^|[_-])vbf|qqtohto", "vbf", {}),
    (r"(?i)h125gggluonfusion|glugluh(to)?|(^|[_-])ggh", "ggf", {}),
    (r"(?i)ttbar|^tt(to|_)|tprimeto|(^|[_-])singletop|(^|[_-])st_t", "top", {}),
    (r"(?i)wprime|wto[lme]nu|(^|[_-])wto", "resonance", dict(seedPdgIds=[24, -24])),
    (r"(?i)zmm|zptomm", "resonance", dict(seedPdgIds=[23, 32], decayPdgIdGroups=[[13, -13]])),
    (r"(?i)zee|zptoee", "resonance", dict(seedPdgIds=[23, 32], decayPdgIdGroups=[[11, -11]])),
    (r"(?i)ztt|zptt|dytotautau", "resonance", dict(seedPdgIds=[23, 32], decayPdgIdGroups=[[15, -15]])),
    (r"(?i)dyto|drell|(^|[_-])z(prime)?to", "resonance", {}),
    (r"(?i)jpsi|psi2s|chic|chib|upsilon|etab", "heavyflavor", dict(seedHadronFlavors=[4])),
    (r"(?i)(^|[_-])b[sdu0c]to|bumixing|lambdab", "heavyflavor", dict(seedHadronFlavors=[5])),
    (r"(?i)sms-|displacedsusy|(^|[_-])susy|glugluto2jets", "full", {}),
    (r"(?i)singlenu|nugun|(^|[_-])nu(e|mu|tau|gun)", "full", {}),
    (r"(?i)^(single|double|triple|four|five|six|ten|eleven|twelve|flat|closeby|ce_)", "gun", {}),
    (r"(?i)qcd|minbias|photonjet", "full", {}),
)


def templateForFragment(name):
    """Return (template_name, overrides) for a generator fragment / gallery label."""
    for pattern, template, overrides in _RULES:
        if re.search(pattern, name):
            ov = dict(overrides)
            if template == "gun":
                ov.setdefault("seedPdgIds", gunSeed(name))
            return template, ov
    return "full", {}


def selectionForFragment(name=None, template=None, **overrides):
    """Resolve a fragment (or an explicit ``template`` name) to a postProcessing
    selection dict, then apply keyword ``overrides`` (full customisation)."""
    if template is None:
        if name is None:
            raise ValueError("selectionForFragment needs a fragment name or a template")
        template, auto = templateForFragment(name)
    else:
        auto = {}
    if template not in TEMPLATES:
        raise KeyError("unknown template %r (known: %s)" % (template, ", ".join(TEMPLATE_NAMES)))
    selection = TEMPLATES[template]()
    selection.update(auto)
    selection.update(overrides)
    return selection


def postProcessingPSet(name=None, template=None, **overrides):
    """``selectionForFragment`` wrapped as a complete ``cms.PSet`` (build-side
    defaults included), ready to drop into a producer's ``postProcessing``."""
    import FWCore.ParameterSet.Config as cms

    s = selectionForFragment(name=name, template=template, **overrides)
    return cms.PSet(
        collapseIntermediateGenParticles=cms.bool(overrides.get("collapseIntermediateGenParticles", True)),
        seedPdgIds=cms.vint32(*s["seedPdgIds"]),
        seedHadronFlavors=cms.vint32(*s["seedHadronFlavors"]),
        seedParentDepth=cms.uint32(s["seedParentDepth"]),
        keepStableSpectators=cms.bool(s["keepStableSpectators"]),
        attachSelectionSources=cms.bool(s["attachSelectionSources"]),
        keepProductionSiblings=cms.bool(s["keepProductionSiblings"]),
        decayPdgIdGroups=cms.VPSet(*[cms.PSet(pdgIds=cms.vint32(*g)) for g in s["decayPdgIdGroups"]]),
        ignoredPdgIds=cms.vint32(*overrides.get("ignoredPdgIds", [])),
        ignoredParticleIds=cms.vuint32(*overrides.get("ignoredParticleIds", [])),
    )


def dumperArgs(name=None, template=None, **overrides):
    """The dumpTruthGraphsFromGENSIMRECO_cfg.py flags for a fragment's selection."""
    s = selectionForFragment(name=name, template=template, **overrides)
    # Empty seedPdgIds means "seed by flavor/decay group, not PDG": omit -s so the
    # dumper keeps it empty (passing -s 0 would force the full-graph escape hatch).
    args = []
    if s["seedPdgIds"]:
        args += ["-s", ",".join(str(p) for p in s["seedPdgIds"])]
    args += ["-d", str(s["seedParentDepth"])]
    if s["seedHadronFlavors"]:
        args += ["-f", ",".join(str(f) for f in s["seedHadronFlavors"])]
    for group in s["decayPdgIdGroups"]:
        args += ["-g", ",".join(str(p) for p in group)]
    args.append("--keepSpectators" if s["keepStableSpectators"] else "--no-keepSpectators")
    args.append("--attachSources" if s["attachSelectionSources"] else "--no-attachSources")
    if s["keepProductionSiblings"]:
        args.append("--keepProductionSiblings")
    return args


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Resolve a generator fragment to its truth-graph selection preset.")
    parser.add_argument("fragment", help="generator fragment cfi name or short gallery label")
    parser.add_argument("--template", default=None, help="force a template instead of auto-resolving")
    parser.add_argument("--dump-args", action="store_true", help="print dumpTruthGraphs... flags (default)")
    parser.add_argument("--name", action="store_true", help="print only the resolved template name")
    args = parser.parse_args()

    if args.name:
        print(args.template or templateForFragment(args.fragment)[0])
    else:
        print(" ".join(dumperArgs(name=args.fragment, template=args.template)))
