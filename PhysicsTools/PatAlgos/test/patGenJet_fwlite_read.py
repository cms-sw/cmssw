#!/usr/bin/env python3
"""Read pat::GenJet collections back in Python and print every field group.

Two inputs are understood:
  - an EDM file (has an "Events" tree): read with FWLite, collection given by --label
    (module[:instance[:process]]), e.g. --label patGenJetsAK4. Needs the producers of phase P.
  - the plain ROOT file written by testGenJet ("t" tree, branch "jets"), which is what this
    script is run on today (no pat::GenJet producer exists yet).

Usage (under cmsenv):
  python3 patGenJet_fwlite_read.py <file.root> [--label module[:instance[:process]]] [--max-events N]
Exit code 0 if at least one jet was read and every field was accessible.
"""
import argparse
import sys

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gSystem.Load("libFWCoreFWLite.so")
ROOT.FWLiteEnabler.enable()
ROOT.gSystem.Load("libDataFormatsPatCandidates.so")


def dump(j, idx, edm=True):
    out = []
    out.append(f"jet {idx}: pt={j.pt():.3f} eta={j.eta():.3f} phi={j.phi():.3f} m={j.mass():.3f} area={j.jetArea():.3f}")
    out.append(f"  specific: em={j.emEnergy():.3f} had={j.hadEnergy():.3f} inv={j.invisibleEnergy():.3f} "
               f"chH={j.chargedHadronEnergy():.3f} neH={j.neutralHadronEnergy():.3f} chEm={j.chargedEmEnergy():.3f} "
               f"neEm={j.neutralEmEnergy():.3f} mu={j.muonEnergy():.3f}")
    out.append(f"  multiplicities: ch={j.chargedMultiplicity()} ne={j.neutralMultiplicity()} "
               f"chH={j.chargedHadronMultiplicity()} neH={j.neutralHadronMultiplicity()} mu={j.muonMultiplicity()}")
    out.append(f"  neutral: jecFactor(Uncorrected)={j.jecFactor('Uncorrected')}")
    sub = [f"{l}:{[(p.id().id(), p.key()) for p in j.subjets(l)]}" for l in j.subjetCollectionNames()]
    # groomedMass dereferences the subjet Ptrs, which needs the event (EDM file only)
    gm = f"{j.groomedMass('SoftDropPuppi'):.3f}" if edm or j.nSubjetCollections() == 0 else "needs event"
    out.append(f"  subjets: {sub} groomedMass(SoftDropPuppi)={gm}")
    br = j.genJetFwdRef().backRef()
    out.append(f"  self link: backRef id={br.id().id() if br.isNonnull() else None} key={br.key() if br.isNonnull() else None}")
    out.append(f"  userFloats: {[(str(n), j.userFloat(str(n))) for n in j.userFloatNames()]}")
    out.append(f"  userInts: {[(str(n), j.userInt(str(n))) for n in j.userIntNames()]}")
    out.append(f"  stored daughters: {j.numberOfDaughters() if j.nSubjetCollections() == 0 else 'flattened (needs event)'}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--label", default="patGenJetsAK4")
    ap.add_argument("--max-events", type=int, default=5)
    a = ap.parse_args()

    f = ROOT.TFile.Open(a.file)
    if not f or f.IsZombie():
        print(f"cannot open {a.file}")
        return 1
    njets = 0
    if f.Get("Events"):
        f.Close()
        from DataFormats.FWLite import Events, Handle
        handle = Handle("std::vector<pat::GenJet>")
        label = tuple(a.label.split(":"))
        for iev, ev in enumerate(Events(a.file)):
            if iev >= a.max_events:
                break
            ev.getByLabel(label, handle)
            for i, j in enumerate(handle.product()):
                print(dump(j, i))
                njets += 1
    else:
        t = f.Get("t")
        for iev in range(min(t.GetEntries(), a.max_events)):
            t.GetEntry(iev)
            for i, j in enumerate(t.jets):
                print(dump(j, i, edm=False))
                njets += 1
    print(f"read {njets} pat::GenJet")
    return 0 if njets > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
