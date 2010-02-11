import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesSequences_cff import *
import copy


zPlots = cms.PSet(
    histograms = cms.VPSet(
    cms.PSet(
    min = cms.untracked.double(0.0),
    max = cms.untracked.double(200.0),
    nbins = cms.untracked.int32(200),
    name = cms.untracked.string("zMass"),
    description = cms.untracked.string("Z mass [GeV/c^{2}]"),
    plotquantity = cms.untracked.string("mass")
    ),
    cms.PSet(
    min = cms.untracked.double(0.0),
    max = cms.untracked.double(200.0),
    nbins = cms.untracked.int32(200),
    name = cms.untracked.string("mu1Pt"),
    description = cms.untracked.string("Highest muon p_{t} [GeV/c]"),
    plotquantity = cms.untracked.string("max(daughter(0).pt,daughter(1).pt)")
    ),
    cms.PSet(
    min = cms.untracked.double(0.0),
    max = cms.untracked.double(200.0),
    nbins = cms.untracked.int32(200),
    name = cms.untracked.string("mu2Pt"),
    description = cms.untracked.string("Lowest muon p_{t} [GeV/c]"),
    plotquantity = cms.untracked.string("min(daughter(0).pt,daughter(1).pt)")
    )
    )
)



# ZMuMu at least 1 HLT + 2 track-iso (Shape)
goodZToMuMuPlots = cms.EDAnalyzer(
    "CandViewHistoAnalyzer",
    zPlots,
    src = cms.InputTag("goodZToMuMuAtLeast1HLT")
)


## half-dirty thing: assuming the label is the same has the collection to plot.... 
## l = len(ZMuMuCategoriesSequences)

## for i in range(0,l):
##      label = ZMuMuCategoriesLabels[i]
##      Plots=copy.deepcopy(goodZToMuMuPlots)
##      Plots.setLabel("Plots")
##      setattr(Plots, "src", cms.InputTag(label))
##      ZMuMuCategoriesSequences[i].__iadd__(Plots)
##      ZMuMuCategoriesSequences[i].setLabel(label)
    


goodZToMuMuSequence.__iadd__(goodZToMuMuPlots)
goodZToMuMuSequence.setLabel("goodZToMuMuAtLeast1HLT")

#ZMuMu 2 HLT + 2  track-iso
goodZToMuMu2HLTPlots = copy.deepcopy(goodZToMuMuPlots)
goodZToMuMu2HLTPlots.src = cms.InputTag("goodZToMuMu2HLT")

goodZToMuMu2HLTSequence.__iadd__(goodZToMuMu2HLTPlots)
goodZToMuMu2HLTSequence.setLabel("goodZToMuMu2HLT")

#ZMuMu 1 HLT + 2  track-iso
goodZToMuMu1HLTPlots = copy.deepcopy(goodZToMuMuPlots)
goodZToMuMu1HLTPlots.src = cms.InputTag("goodZToMuMu1HLT")

goodZToMuMu1HLTSequence.__iadd__(goodZToMuMu1HLTPlots)


#ZMuMu at least 1 HLT + at least 1 NON track-iso
nonIsolatedZToMuMuPlots = copy.deepcopy(goodZToMuMuPlots)
nonIsolatedZToMuMuPlots.src = cms.InputTag("nonIsolatedZToMuMuAtLeast1HLT")

nonIsolatedZToMuMuSequence.__iadd__(nonIsolatedZToMuMuPlots)

#ZMuMu at least 1 HLT + 1 NON track-iso
oneNonIsolatedZToMuMuPlots = copy.deepcopy(goodZToMuMuPlots)
oneNonIsolatedZToMuMuPlots.src = cms.InputTag("oneNonIsolatedZToMuMuAtLeast1HLT")

oneNonIsolatedZToMuMuSequence.__iadd__(oneNonIsolatedZToMuMuPlots) 


#ZMuMu at least 1 HLT + 2 NON track-iso
twoNonIsolatedZToMuMuPlots = copy.deepcopy(goodZToMuMuPlots)
twoNonIsolatedZToMuMuPlots.src = cms.InputTag("twoNonIsolatedZToMuMuAtLeast1HLT")

twoNonIsolatedZToMuMuSequence.__iadd__(twoNonIsolatedZToMuMuPlots) 

#ZMuSta First HLT + 2  track-iso
goodZToMuMuOneStandAloneMuonPlots = copy.deepcopy(goodZToMuMuPlots)
goodZToMuMuOneStandAloneMuonPlots.src = cms.InputTag("goodZToMuMuOneStandAloneMuonFirstHLT")

goodZToMuMuOneStandAloneMuonSequence.__iadd__(goodZToMuMuOneStandAloneMuonPlots)


#ZMuTk First HLT + 2  track-iso
goodZToMuMuOneTrackPlots = copy.deepcopy(goodZToMuMuPlots)
goodZToMuMuOneTrackPlots.src = cms.InputTag("goodZToMuMuOneTrackFirstHLT")

goodZToMuMuOneTrackSequence.__iadd__(goodZToMuMuOneTrackPlots)

#ZMuMu same charge
goodZToMuMuSameChargeAtLeast1HLTPlots = copy.deepcopy(goodZToMuMuPlots)
goodZToMuMuSameChargeAtLeast1HLTPlots.src = cms.InputTag("goodZToMuMuSameChargeAtLeast1HLT")

goodZToMuMuSameChargeSequence.__iadd__(goodZToMuMuSameChargeAtLeast1HLTPlots)

goodZToMuMuSameCharge2HLTPlots = copy.deepcopy(goodZToMuMuPlots)
goodZToMuMuSameCharge2HLTPlots.src = cms.InputTag("goodZToMuMuSameCharge2HLT")

goodZToMuMuSameCharge2HLTSequence.__iadd__(goodZToMuMuSameCharge2HLTPlots)

goodZToMuMuSameCharge1HLTPlots = copy.deepcopy(goodZToMuMuPlots)
goodZToMuMuSameCharge1HLTPlots.src = cms.InputTag("goodZToMuMuSameCharge1HLT")

goodZToMuMuSameCharge1HLTSequence.__iadd__(goodZToMuMuSameCharge1HLTPlots)
