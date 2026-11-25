#!/usr/bin/env python3
# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

"""One analysis per selection preset, written with graphTools.

Each function takes a TruthGraphView and prints one line per object of interest. The
physics question follows the preset of truthGraphSelections.py: what the preset seeds on
is what the example starts from.

    presetExamples.py --preset top   step3.root
    presetExamples.py --preset all   truthlogicalgraph_run1_lumi1_event7.json

The C++ twin is TruthGraphPresetExamples.cc, one function per preset with the same names.
"""

import argparse
import math

from PhysicsTools.TruthInfo.graphTools import TruthGraphView, eventGraphs

LEPTONS = (11, 13, 15)
NEUTRINOS = (12, 14, 16)


def mass(p4):
    return math.sqrt(max(p4[3] ** 2 - p4[0] ** 2 - p4[1] ** 2 - p4[2] ** 2, 0.0))


def pt(p4):
    return math.hypot(p4[0], p4[1])


def eta(p4):
    p = math.sqrt(p4[0] ** 2 + p4[1] ** 2 + p4[2] ** 2)
    return 0.5 * math.log((p + p4[2]) / (p - p4[2])) if p > abs(p4[2]) else float("inf")


def add(*p4s):
    return tuple(sum(c) for c in zip(*p4s))


def firstChildWithPdgId(graph, particle, pdgIds):
    for child in graph.children(graph.lastCopy(particle)):
        if abs(graph.pdgId(child)) in pdgIds:
            return child
    return None


def decayMode(graph, boson):
    """leptonic, hadronic or none, from the children of a W or a Z. A tau counts as a
    lepton here, so W -> tau nu is leptonic whatever the tau does next."""
    pdgIds = [abs(graph.pdgId(c)) for c in graph.children(graph.lastCopy(boson))]
    if any(p in LEPTONS for p in pdgIds):
        return "leptonic"
    if any(p <= 6 for p in pdgIds):
        return "hadronic"
    return "none"


# --- gun: each gun particle is its own signal ------------------------------------------
def gun(graph):
    for seed in graph.particlesOfLevel("signal"):
        # The gun particle and everything below it: a stable gun particle is its own product.
        subgraph = [seed] + graph.descendants(seed)
        products = [i for i in subgraph if graph.isAtLevel(i, "reconstructableFromSignal")]
        atCalo = [i for i in subgraph if graph.isAtLevel(i, "caloBoundary")]
        print(f"gun {graph.pdgId(seed):6d} E {graph.p4(seed)[3]:8.2f} GeV: {len(products)} reconstructable products, "
              f"{len(atCalo)} descendants reach the calorimeter")


# --- resonance: the boson and its leptonic legs -------------------------------------------
def resonance(graph):
    for z in graph.particlesOfLevel("signal"):
        legs = [c for c in graph.children(graph.lastCopy(z)) if abs(graph.pdgId(c)) in LEPTONS]
        if len(legs) != 2:
            print(f"resonance {graph.pdgId(z)}: decay mode {decayMode(graph, z)}")
            continue
        print(f"resonance {graph.pdgId(z)} -> {graph.pdgId(legs[0])} {graph.pdgId(legs[1])}: "
              f"m(ll) {mass(add(graph.p4(legs[0]), graph.p4(legs[1]))):7.2f} GeV, "
              f"generator mass {mass(graph.p4(z)):7.2f} GeV")


# --- vbf: the Higgs and the two tagging quarks ---------------------------------------------
def vbf(graph):
    higgs = graph.particlesOfLevel("signal")
    tagging = [i for i in graph.particlesOfLevel("partonJets") if graph.isSignal(i)]
    if len(tagging) < 2:
        print(f"vbf: {len(higgs)} Higgs, {len(tagging)} tagging partons")
        return
    tagging.sort(key=lambda i: -pt(graph.p4(i)))
    j1, j2 = tagging[0], tagging[1]
    print(f"vbf: {len(higgs)} Higgs, tagging partons {graph.pdgId(j1)} {graph.pdgId(j2)}: "
          f"m(jj) {mass(add(graph.p4(j1), graph.p4(j2))):7.1f} GeV, "
          f"|delta eta| {abs(eta(graph.p4(j1)) - eta(graph.p4(j2))):5.2f}")


# --- ggf: the Higgs and what the detector can see of it -----------------------------------
def ggf(graph):
    for higgs in graph.particlesOfLevel("signal"):
        products = [i for i in graph.descendants(higgs) if graph.isAtLevel(i, "reconstructableFromSignal")]
        visible = sum(graph.p4(i)[3] for i in products if abs(graph.pdgId(i)) not in NEUTRINOS)
        print(f"ggf: Higgs E {graph.p4(higgs)[3]:8.2f} GeV -> {len(products)} reconstructable products, "
              f"visible fraction {visible / graph.p4(higgs)[3]:5.3f}")


# --- vh: the Higgs and the boson produced with it ------------------------------------------
def vh(graph):
    for higgs in graph.particlesOfLevel("signal"):
        siblings = [s for v in graph.productionVertices(higgs) for s in graph.outgoingParticles(v)
                    if s != higgs and abs(graph.pdgId(s)) in (23, 24)]
        for boson in siblings:
            print(f"vh: Higgs with {graph.pdgId(boson)} pt {pt(graph.p4(boson)):7.2f} GeV, "
                  f"boson decay {decayMode(graph, boson)}")


# --- top: the two tops, their b and W, and the event class ---------------------------------
def top(graph):
    modes = []
    for t in graph.particlesOfLevel("signal"):
        b = firstChildWithPdgId(graph, t, (5,))
        w = firstChildWithPdgId(graph, t, (24,))
        mode = decayMode(graph, w) if w is not None else "none"
        modes.append(mode)
        print(f"top {graph.pdgId(t):3d}: b {'yes' if b is not None else 'no'}, W {mode}, "
              f"{len(graph.descendants(t))} descendants")
    leptonic = modes.count("leptonic")
    print("top event class:", {0: "all hadronic", 1: "semileptonic", 2: "dilepton"}.get(leptonic, "other"))


# --- singletop: the top and its production partner ----------------------------------------
def singletop(graph):
    for t in graph.particlesOfLevel("signal"):
        partners = [s for v in graph.productionVertices(t) for s in graph.outgoingParticles(v) if s != t]
        for partner in partners:
            print(f"singletop: top with partner {graph.pdgId(partner)} pt {pt(graph.p4(partner)):7.2f} GeV")


# --- diboson: the bosons, their modes and their mass --------------------------------------
def diboson(graph):
    bosons = [i for i in graph.particlesOfLevel("signal") if abs(graph.pdgId(i)) in (23, 24)]
    if len(bosons) >= 2:
        print(f"diboson: m(VV) {mass(add(*[graph.p4(b) for b in bosons[:2]])):7.1f} GeV")
    for boson in bosons:
        print(f"  {graph.pdgId(boson):4d} pt {pt(graph.p4(boson)):7.2f} GeV, decay {decayMode(graph, boson)}")


# --- heavyflavor: the b hadrons and their flight -----------------------------------------
def heavyflavor(graph):
    for hadron in graph.particlesOfLevel("bHadrons"):
        production = graph.productionVertices(hadron)
        decay = graph.decayVertices(hadron)
        flight = None
        if production and decay:
            x0, x1 = graph.vertex(production[0])["x4"], graph.vertex(decay[0])["x4"]
            flight = math.sqrt(sum((a - b) ** 2 for a, b in zip(x0[:3], x1[:3])))
        charm = [i for i in graph.descendants(hadron) if graph.isAtLevel(i, "cHadrons")]
        print(f"heavyflavor: {graph.pdgId(hadron):6d} pt {pt(graph.p4(hadron)):7.2f} GeV, "
              f"flight {'n/a' if flight is None else f'{flight:6.3f} cm'}, "
              f"{len(charm)} charm hadron{'s' if len(charm) != 1 else ''} below")


# --- full: the whole event, signal and pile-up apart ---------------------------------------
def full(graph):
    print(graph.summary())
    for side, keep in (("signal", graph.isSignal), ("pileup", graph.isFromPileup)):
        members = [i for i in graph.particlesOfLevel("reconstructableFinalState") if keep(i)]
        energy = sum(graph.p4(i)[3] for i in members)
        print(f"full: {side} reconstructable final state {len(members)} objects, {energy:9.1f} GeV, "
              f"{sum(1 for i in members if not graph.hasMomentum(i))} without momentum")


PRESETS = {
    "gun": gun, "resonance": resonance, "vbf": vbf, "ggf": ggf, "vh": vh,
    "top": top, "singletop": singletop, "diboson": diboson, "heavyflavor": heavyflavor, "full": full,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="all", choices=["all"] + list(PRESETS))
    parser.add_argument("-n", "--maxEvents", type=int, default=-1)
    parser.add_argument("inputs", nargs="+", help="an EDM file, or one or more JSON dumps")
    args = parser.parse_args()

    if args.inputs[0].endswith(".root"):
        graphs = eventGraphs(args.inputs[0], maxEvents=args.maxEvents)
    else:
        graphs = (TruthGraphView.fromJson(path) for path in args.inputs)

    names = list(PRESETS) if args.preset == "all" else [args.preset]
    for graph in graphs:
        print(f"== event {graph.eventId}")
        for name in names:
            PRESETS[name](graph)


if __name__ == "__main__":
    main()
