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
goodZToMuMuPlotsLoose = cms.EDAnalyzer(
    "CandViewHistoAnalyzer",
    zPlots,
    src = cms.InputTag("goodZToMuMuAtLeast1HLTLoose")
)

goodZToMuMuPlots = cms.EDAnalyzer(
    "CandViewHistoAnalyzer",
    zPlots,
    src = cms.InputTag("goodZToMuMuAtLeast1HLT")
)



## #### plot for loose cuts


## goodZToMuMuSequence.__iadd__(goodZToMuMuPlots)
## goodZToMuMuSequence.setLabel("goodZToMuMuAtLeast1HLT")

## #ZMuMu 2 HLT + 2  track-iso
## goodZToMuMu2HLTPlots = copy.deepcopy(goodZToMuMuPlots)
## goodZToMuMu2HLTPlots.src = cms.InputTag("goodZToMuMu2HLT")

## goodZToMuMu2HLTSequence.__iadd__(goodZToMuMu2HLTPlots)
## goodZToMuMu2HLTSequence.setLabel("goodZToMuMu2HLT")

## #ZMuMu 1 HLT + 2  track-iso
## goodZToMuMu1HLTPlots = copy.deepcopy(goodZToMuMuPlots)
## goodZToMuMu1HLTPlots.src = cms.InputTag("goodZToMuMu1HLT")

## goodZToMuMu1HLTSequence.__iadd__(goodZToMuMu1HLTPlots)


## #ZMuMu at least 1 HLT + at least 1 NON track-iso
## nonIsolatedZToMuMuPlots = copy.deepcopy(goodZToMuMuPlots)
## nonIsolatedZToMuMuPlots.src = cms.InputTag("nonIsolatedZToMuMuAtLeast1HLT")

## nonIsolatedZToMuMuSequence.__iadd__(nonIsolatedZToMuMuPlots)

## #ZMuMu at least 1 HLT + 1 NON track-iso
## oneNonIsolatedZToMuMuPlots = copy.deepcopy(goodZToMuMuPlots)
## oneNonIsolatedZToMuMuPlots.src = cms.InputTag("oneNonIsolatedZToMuMuAtLeast1HLT")

## oneNonIsolatedZToMuMuSequence.__iadd__(oneNonIsolatedZToMuMuPlots) 


## #ZMuMu at least 1 HLT + 2 NON track-iso
## twoNonIsolatedZToMuMuPlots = copy.deepcopy(goodZToMuMuPlots)
## twoNonIsolatedZToMuMuPlots.src = cms.InputTag("twoNonIsolatedZToMuMuAtLeast1HLT")

## twoNonIsolatedZToMuMuSequence.__iadd__(twoNonIsolatedZToMuMuPlots) 

## #ZMuSta First HLT + 2  track-iso
## goodZToMuMuOneStandAloneMuonPlots = copy.deepcopy(goodZToMuMuPlots)
## goodZToMuMuOneStandAloneMuonPlots.src = cms.InputTag("goodZToMuMuOneStandAloneMuonFirstHLT")

## goodZToMuMuOneStandAloneMuonSequence.__iadd__(goodZToMuMuOneStandAloneMuonPlots)


## #ZMuTk First HLT + 2  track-iso
## goodZToMuMuOneTrackPlots = copy.deepcopy(goodZToMuMuPlots)
## goodZToMuMuOneTrackPlots.src = cms.InputTag("goodZToMuMuOneTrackFirstHLT")

## goodZToMuMuOneTrackSequence.__iadd__(goodZToMuMuOneTrackPlots)

## #ZMuMu same charge
## goodZToMuMuSameChargeAtLeast1HLTPlots = copy.deepcopy(goodZToMuMuPlots)
## goodZToMuMuSameChargeAtLeast1HLTPlots.src = cms.InputTag("goodZToMuMuSameChargeAtLeast1HLT")

## goodZToMuMuSameChargeSequence.__iadd__(goodZToMuMuSameChargeAtLeast1HLTPlots)

## goodZToMuMuSameCharge2HLTPlots = copy.deepcopy(goodZToMuMuPlots)
## goodZToMuMuSameCharge2HLTPlots.src = cms.InputTag("goodZToMuMuSameCharge2HLT")

## goodZToMuMuSameCharge2HLTSequence.__iadd__(goodZToMuMuSameCharge2HLTPlots)

## goodZToMuMuSameCharge1HLTPlots = copy.deepcopy(goodZToMuMuPlots)
## goodZToMuMuSameCharge1HLTPlots.src = cms.InputTag("goodZToMuMuSameCharge1HLT")

## goodZToMuMuSameCharge1HLTSequence.__iadd__(goodZToMuMuSameCharge1HLTPlots)



#### plot for tight cuts


goodZToMuMuPath.__iadd__(goodZToMuMuPlots)
goodZToMuMuPath.setLabel("goodZToMuMuAtLeast1HLT")

#ZMuMu 2 HLT + 2  track-iso
goodZToMuMu2HLTPlots = copy.deepcopy(goodZToMuMuPlots)
goodZToMuMu2HLTPlots.src = cms.InputTag("goodZToMuMu2HLT")

goodZToMuMu2HLTPath.__iadd__(goodZToMuMu2HLTPlots)
goodZToMuMu2HLTPath.setLabel("goodZToMuMu2HLT")

#ZMuMu 1 HLT + 2  track-iso
goodZToMuMu1HLTPlots= copy.deepcopy(goodZToMuMuPlots)
goodZToMuMu1HLTPlots.src = cms.InputTag("goodZToMuMu1HLT")

goodZToMuMu1HLTPath.__iadd__(goodZToMuMu1HLTPlots)


##### plot for AB and BB region
goodZToMuMuAB1HLTPlots= copy.deepcopy(goodZToMuMuPlots)
goodZToMuMuAB1HLTPlots.src = cms.InputTag("goodZToMuMuAB1HLT")
goodZToMuMuAB1HLTPath.__iadd__(goodZToMuMuAB1HLTPlots)

goodZToMuMuBB2HLTPlots= copy.deepcopy(goodZToMuMuPlots)
goodZToMuMuBB2HLTPlots.src = cms.InputTag("goodZToMuMuBB2HLT")
goodZToMuMuBB2HLTPath.__iadd__(goodZToMuMuBB2HLTPlots)



#ZMuMu at least 1 HLT + at least 1 NON track-iso
nonIsolatedZToMuMuPlots = copy.deepcopy(goodZToMuMuPlots)
nonIsolatedZToMuMuPlots.src = cms.InputTag("nonIsolatedZToMuMuAtLeast1HLT")

nonIsolatedZToMuMuPath.__iadd__(nonIsolatedZToMuMuPlots)

#ZMuMu at least 1 HLT + 1 NON track-iso
oneNonIsolatedZToMuMuPlots = copy.deepcopy(goodZToMuMuPlots)
oneNonIsolatedZToMuMuPlots.src = cms.InputTag("oneNonIsolatedZToMuMuAtLeast1HLT")

oneNonIsolatedZToMuMuPath.__iadd__(oneNonIsolatedZToMuMuPlots) 


#ZMuMu at least 1 HLT + 2 NON track-iso
twoNonIsolatedZToMuMuPlots = copy.deepcopy(goodZToMuMuPlots)
twoNonIsolatedZToMuMuPlots.src = cms.InputTag("twoNonIsolatedZToMuMuAtLeast1HLT")

twoNonIsolatedZToMuMuPath.__iadd__(twoNonIsolatedZToMuMuPlots) 

#ZMuSta global HLT + 2  track-iso
goodZToMuMuOneStandAloneMuonPlots = copy.deepcopy(goodZToMuMuPlots)
goodZToMuMuOneStandAloneMuonPlots.src = cms.InputTag("goodZToMuMuOneStandAloneMuonFirstHLT")

goodZToMuMuOneStandAloneMuonPath.__iadd__(goodZToMuMuOneStandAloneMuonPlots)


#ZMuTk First HLT + 2  track-iso
goodZToMuMuOneTrackPlots = copy.deepcopy(goodZToMuMuPlots)
goodZToMuMuOneTrackPlots.src = cms.InputTag("goodZToMuMuOneTrackFirstHLT")

goodZToMuMuOneTrackPath.__iadd__(goodZToMuMuOneTrackPlots)


#ZMuTkMu global HLT + 2  track-iso
goodZToMuMuOneTrackerMuonPlots = copy.deepcopy(goodZToMuMuPlots)
goodZToMuMuOneTrackerMuonPlots.src = cms.InputTag("goodZToMuMuOneTrackerMuonFirstHLT")

goodZToMuMuOneTrackerMuonPath.__iadd__(goodZToMuMuOneTrackerMuonPlots)






#ZMuMu same charge
goodZToMuMuSameChargeAtLeast1HLTPlots = copy.deepcopy(goodZToMuMuPlots)
goodZToMuMuSameChargeAtLeast1HLTPlots.src = cms.InputTag("goodZToMuMuSameChargeAtLeast1HLT")

goodZToMuMuSameChargePath.__iadd__(goodZToMuMuSameChargeAtLeast1HLTPlots)

goodZToMuMuSameCharge2HLTPlots = copy.deepcopy(goodZToMuMuPlots)
goodZToMuMuSameCharge2HLTPlots.src = cms.InputTag("goodZToMuMuSameCharge2HLT")

goodZToMuMuSameCharge2HLTPath.__iadd__(goodZToMuMuSameCharge2HLTPlots)

goodZToMuMuSameCharge1HLTPlots = copy.deepcopy(goodZToMuMuPlots)
goodZToMuMuSameCharge1HLTPlots.src = cms.InputTag("goodZToMuMuSameCharge1HLT")

goodZToMuMuSameCharge1HLTPath.__iadd__(goodZToMuMuSameCharge1HLTPlots)
