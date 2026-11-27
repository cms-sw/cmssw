#!/usr/bin/env python3
"""The collections made by addPatGenJetCollection (patGenJet_test_tool_cfg.py) equal the ones of patGenJets_cff, jet by jet.

Usage: python3 patGenJet_check_tool_output.py tool.root
  patGenJetsAK4Tool  == patGenJetsAK4                   p4, daughters
  patGenJetsAK4                                         genJetFwdRef().backRef() points into ak4GenJetsNoNu (same product, pt of the entry)
  packedPatGenJetsAK8PuppiTool == packedPatGenJetsAK8    p4, daughters, user floats (tau1-4, soft-drop mass, ECFs),
                                                         subjets(SoftDropPuppi) keys, groomed mass
  packedPatGenJetsAK8PuppiTool:SubJets == packedPatGenJetsAK8:SubJets   p4, user floats (ECFs, tau under the other name)
Exit code 0 if everything is equal.
"""
import sys
import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gSystem.Load("libFWCoreFWLite.so")
ROOT.FWLiteEnabler.enable()
ROOT.gSystem.Load("libDataFormatsPatCandidates.so")
from DataFormats.FWLite import Events, Handle

H = Handle("std::vector<pat::GenJet>")
G = Handle("std::vector<reco::GenJet>")
events = Events(sys.argv[1])
bad = []
count = {"ak4": 0, "fat": 0, "sub": 0, "ref": 0}

def get(ev, label):
    ev.getByLabel(label, H) if ":" not in label else ev.getByLabel(*label.split(":"), H)
    return [j for j in H.product()]

def same_jet(a, b, what, floats=()):
    for f in ("pt", "eta", "phi", "mass"):
        if getattr(a, f)() != getattr(b, f)():
            bad.append("%s %s differs" % (what, f))
    if [p.key() for p in a.daughterPtrVector()] != [p.key() for p in b.daughterPtrVector()]:
        bad.append("%s daughters differ" % what)
    for fa, fb in floats:
        if not a.hasUserFloat(fa) or not b.hasUserFloat(fb) or a.userFloat(fa) != b.userFloat(fb):
            bad.append("%s userFloat %s / %s: %s" % (what, fa, fb, (a.userFloat(fa) if a.hasUserFloat(fa) else None, b.userFloat(fb) if b.hasUserFloat(fb) else None)))

TAU = ["NjettinessAK8Puppi:tau%d" % i for i in (1, 2, 3, 4)]
ECF = ["nb%d%s:ecf%s" % (b, "AK8PuppiSoftDrop", n) for b in (1, 2) for n in ("N2", "N3")]
SECF = ["nb%d%s:ecf%s" % (b, "AK8PuppiSoftDropSubjets", n) for b in (1, 2) for n in ("N2", "N3")]
def check_back_refs(ev, jets, what):
    """backRef of genJetFwdRef = the entry of ak4GenJetsNoNu the jet was made from (same index in pt order, same pt)."""
    ev.getByLabel("ak4GenJetsNoNu", G)
    gen = G.product()
    for i, j in enumerate(jets):
        l = j.genJetFwdRef()
        b = l.backRef()
        # b.get() resolves the product ID through the event: it is a gen jet of ak4GenJetsNoNu (the only product of that type in the file)
        if b.isNull() or b.key() >= len(gen) or b.get() is None or abs(b.get().pt() - j.pt()) > 1e-4 * max(1., j.pt()):
            bad.append("%s[%d] genJetFwdRef back ref not in ak4GenJetsNoNu" % (what, i)); continue
        if abs(gen[b.key()].pt() - j.pt()) > 1e-4 * max(1., j.pt()):
            bad.append("%s[%d] back ref pt %f != jet pt %f" % (what, i, gen[b.key()].pt(), j.pt()))
        if l.ref().isNull() or l.ref().id() != b.id() or l.ref().key() != b.key():
            bad.append("%s[%d] forward ref != back ref (no genJetForwardMap)" % (what, i))
        count["ref"] += 1

for ev in events:
    a, b = get(ev, "patGenJetsAK4Tool"), get(ev, "patGenJetsAK4")
    check_back_refs(ev, b, "patGenJetsAK4")
    if len(a) != len(b):
        bad.append("AK4 sizes %d %d" % (len(a), len(b)))
    for i, (x, y) in enumerate(zip(a, b)):
        same_jet(x, y, "AK4[%d]" % i); count["ak4"] += 1
    fa, fb = get(ev, "packedPatGenJetsAK8PuppiTool"), get(ev, "packedPatGenJetsAK8")
    if len(fa) != len(fb):
        bad.append("fat sizes %d %d" % (len(fa), len(fb)))
    for i, (x, y) in enumerate(zip(fa, fb)):
        floats = [(t, t) for t in TAU] + [("groomedMassAK8Puppi", "ak8PFJetsPuppiSoftDropMass")]
        if x.hasUserFloat(ECF[0]) != y.hasUserFloat(ECF[0]):
            bad.append("fat[%d] ECF presence differs" % i)
        elif y.hasUserFloat(ECF[0]):
            floats += [(t, t) for t in ECF]
        same_jet(x, y, "fat[%d]" % i, floats)
        sx, sy = x.subjets("SoftDropPuppi"), y.subjets("SoftDropPuppi")
        if [p.key() for p in sx] != [p.key() for p in sy]:
            bad.append("fat[%d] subjet keys differ" % i)
        if x.groomedMass("SoftDropPuppi") != y.groomedMass("SoftDropPuppi"):
            bad.append("fat[%d] groomed mass differs" % i)
        count["fat"] += 1
    sa, sb = get(ev, "packedPatGenJetsAK8PuppiTool:SubJets"), get(ev, "packedPatGenJetsAK8:SubJets")
    if len(sa) != len(sb):
        bad.append("sub sizes %d %d" % (len(sa), len(sb)))
    for i, (x, y) in enumerate(zip(sa, sb)):
        floats = [(t, t) for t in SECF] + [("NjettinessAK8PuppiSubjets:tau%d" % k, "NjettinessAK8Subjets:tau%d" % k) for k in (1, 2, 3, 4)]
        same_jet(x, y, "sub[%d]" % i, floats); count["sub"] += 1
print("jets compared:", count)
if min(count.values()) == 0:
    bad.append("nothing compared in some collection")
if bad:
    print("FAILED (%d):" % len(bad)); print("\n".join("  " + b for b in bad[:30])); sys.exit(1)
print("check_tool_output: OK")
