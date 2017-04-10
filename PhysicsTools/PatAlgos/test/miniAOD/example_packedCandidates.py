# import ROOT in batch mode
import sys
oldargv = sys.argv[:]
sys.argv = [ '-b-' ]
import ROOT
ROOT.gROOT.SetBatch(True)
sys.argv = oldargv

# load FWLite C++ libraries
ROOT.gSystem.Load("libFWCoreFWLite.so");
ROOT.gSystem.Load("libDataFormatsFWLite.so");
ROOT.FWLiteEnabler.enable()

# load FWlite python libraries
from DataFormats.FWLite import Handle, Events

# define deltaR
from math import hypot, pi
def deltaR(a,b):
    dphi = abs(a.phi()-b.phi());
    if dphi < pi: dphi = 2*pi-dphi
    return hypot(a.eta()-b.eta(),dphi)

muons, muonLabel = Handle("std::vector<pat::Muon>"), "slimmedMuons"
electrons, electronLabel = Handle("std::vector<pat::Electron>"), "slimmedElectrons"
jets, jetLabel = Handle("std::vector<pat::Jet>"), "slimmedJets"
pfs, pfLabel = Handle("std::vector<pat::PackedCandidate>"), "packedPFCandidates"

# open file (you can use 'edmFileUtil -d /store/whatever.root' to get the physical file name)
events = Events("patMiniAOD_standard.root")

for iev,event in enumerate(events):
    event.getByLabel(muonLabel, muons)
    event.getByLabel(electronLabel, electrons)
    event.getByLabel(pfLabel, pfs)
    event.getByLabel(jetLabel, jets)

    print "\nEvent %d: run %6d, lumi %4d, event %12d" % (iev,event.eventAuxiliary().run(), event.eventAuxiliary().luminosityBlock(),event.eventAuxiliary().event())

    # Let's compute lepton PF Isolation with R=0.2, 0.5 GeV threshold on neutrals, and deltaBeta corrections
    leps  = [ p for p in muons.product() ] + [ p for p in electrons.product() ]
    for lep in leps:
        # skip those below 5 GeV, which we don't care about
        if lep.pt() < 5: continue
        # initialize sums
        charged = 0
        neutral = 0
        pileup  = 0
        # now get a list of the PF candidates used to build this lepton, so to exclude them
        footprint = set()
        for i in xrange(lep.numberOfSourceCandidatePtrs()):
            footprint.add(lep.sourceCandidatePtr(i).key()) # the key is the index in the pf collection
        # now loop on pf candidates
        for ipf,pf in enumerate(pfs.product()):
            if deltaR(pf,lep) < 0.2:
                # pfcandidate-based footprint removal
                if ipf in footprint: continue
                # add up
                if (pf.charge() == 0):
                    if pf.pt() > 0.5: neutral += pf.pt()
                elif pf.fromPV() >= 2:
                    charged += pf.pt()
                else:
                    if pf.pt() > 0.5: pileup += pf.pt()
        # do deltaBeta
        iso = charged + max(0, neutral-0.5*pileup)
        print "%-8s of pt %6.1f, eta %+4.2f: relIso = %5.2f" % (
                    "muon" if abs(lep.pdgId())==13 else "electron",
                    lep.pt(), lep.eta(), iso/lep.pt())

    # Let's compute the fraction of charged pt from particles with dz < 0.1 cm
    for i,j in enumerate(jets.product()):
        if j.pt() < 40 or abs(j.eta()) > 2.4: continue
        sums = [0,0]
        for id in xrange(j.numberOfDaughters()):
            dau = j.daughter(id)
            if (dau.charge() == 0): continue
            sums[ abs(dau.dz())<0.1 ] += dau.pt()
        sum = sums[0]+sums[1]
        print "Jet with pt %6.1f, eta %+4.2f, beta(0.1) = %+5.3f, pileup mva disc %+.2f" % (
                j.pt(),j.eta(), sums[1]/sum if sum else 0, j.userFloat("pileupJetId:fullDiscriminant"))

    # Let's check the calorimeter response for hadrons (after PF hadron calibration)
    for i,j in enumerate(pfs.product()):
        if not j.isIsolatedChargedHadron(): continue
        print "Isolated charged hadron candidate with pt %6.1f, eta %+4.2f, calo/track energy = %+5.3f, hcal/calo energy %+5.3f" % (
                j.pt(),j.eta(), j.rawCaloFraction(), j.hcalFraction())
    
    if iev > 10: break
