#!/usr/bin/env python3
"""embedGenCandidates=True on patGenJetsAK4 (patGenJet_test_tool_cfg.py embed=True): numbers for the decision.

Usage: python3 patGenJet_embed_stats.py without.root with.root
  - stored gen candidates over all AK4 jets; fraction whose forward Ptr is in another product than the back Ptr
    (forward.id() != back.id(): the particle is in prunedGenParticles); fraction with forward == back (no partner in the chain)
  - size of the patGenJetsAK4 branch per event (compressed and uncompressed) in both files
Exit code 0 if the checks pass.
"""
import sys
import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gSystem.Load("libFWCoreFWLite.so")
ROOT.FWLiteEnabler.enable()
ROOT.gSystem.Load("libDataFormatsPatCandidates.so")
from DataFormats.FWLite import Events, Handle

def branch_size(fname):
    f = ROOT.TFile.Open(fname)
    t = f.Get("Events")
    n = t.GetEntries()
    br = [b for b in t.GetListOfBranches() if b.GetName().startswith("patGenJets_patGenJetsAK4__")]
    assert len(br) == 1, [b.GetName() for b in t.GetListOfBranches()]
    # a split branch: sum the sub-branches
    sizes = [0, 0]
    def walk(x):
        sizes[0] += x.GetZipBytes()
        sizes[1] += x.GetTotBytes()
        for sub in x.GetListOfBranches():
            walk(sub)
    walk(br[0])
    return n, sizes[0], sizes[1]

nw, zw, tw = branch_size(sys.argv[1])
ne, ze, te = branch_size(sys.argv[2])
H = Handle("std::vector<pat::GenJet>")
G = Handle("std::vector<reco::GenParticle>")
events = Events(sys.argv[2])
bad = []
nJets = nCand = nFwd = nSame = 0
for ev in events:
    ev.getByLabel("patGenJetsAK4", H)
    ev.getByLabel("prunedGenParticles", G)
    pruned = G.product()
    for j in H.product():
        nJets += 1
        for fp in j.genCandidatesFwdPtr():
            nCand += 1
            if fp.backPtr().isNull():
                bad.append("null back Ptr")
            if fp.ptr().id() != fp.backPtr().id():
                nFwd += 1
                if fp.ptr().key() >= len(pruned):
                    bad.append("forward key out of range")
            else:
                nSame += 1
                if fp.ptr().key() != fp.backPtr().key():
                    bad.append("forward in the back product with another key")
print("AK4 jets: %d  stored gen candidates: %d (%.2f per jet)" % (nJets, nCand, nCand / max(1, nJets)))
print("forward Ptr in prunedGenParticles (forward.id != back.id): %d  fraction %.4f" % (nFwd, nFwd / max(1, nCand)))
print("forward == back (no partner in the chain): %d  fraction %.4f" % (nSame, nSame / max(1, nCand)))
print("patGenJetsAK4 branch, %d events:" % nw)
print("  without embedGenCandidates: compressed %.0f bytes/event, uncompressed %.0f bytes/event" % (zw / nw, tw / nw))
print("  with    embedGenCandidates: compressed %.0f bytes/event, uncompressed %.0f bytes/event" % (ze / ne, te / ne))
if nCand == 0:
    bad.append("no gen candidates stored")
if bad:
    print("FAILED:", sorted(set(bad))); sys.exit(1)
print("embed_stats: OK")
