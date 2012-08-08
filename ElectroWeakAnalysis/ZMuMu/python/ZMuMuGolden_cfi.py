import FWCore.ParameterSet.Config as cms

import copy

###################################################
#              muons for ZMuMu                    #    
###################################################

goodGlobalMuons = cms.EDFilter("MuonViewRefSelector",
  src = cms.InputTag("muons"),
  cut = cms.string('isGlobalMuon = 1 & isTrackerMuon = 1 &  pt > 20 & abs(eta)<2.1 & isolationR03().sumPt<3.0 & abs(globalTrack().dxy)<0.2 & globalTrack().hitPattern().numberOfValidTrackerHits>10'),
  filter = cms.bool(True)                                
)

###################################################
#              combiner module                    #    
###################################################

zmmCands = cms.EDFilter("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(True),
    cut = cms.string('mass > 60 & mass<120 &  charge=0'),
    decay = cms.string("goodGlobalMuons@+ goodGlobalMuons@-")
)


# dimuon filter
dimuonsFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("zmmCands"),
    minNumber = cms.uint32(1)
)

### trigger filter: selection of the events which have fired the HLT trigger path given. You may want to use it or to duisegard at all add a trigger match or not....



import HLTrigger.HLTfilters.hltHighLevel_cfi

dimuonsHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
# Add this to access 8E29 menu
dimuonsHLTFilter.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
# for 1E31 menu
#dimuonsHLTFilter.HLTPaths = ["HLT_Mu9", "HLT_DoubleMu3"]
dimuonsHLTFilter.HLTPaths = ["HLT_Mu9"]




##################################################
###    trigger mathching, optional         #######
##################################################

zSelection = cms.PSet(
## cut already implemented, but one could add more (e.g. massMin, massMax,... change the pt or eta cut....)
    cut = cms.string("charge = 0 & daughter(0).pt > 20 & daughter(1).pt > 20 & abs(daughter(0).eta)<2.1 & abs(daughter(1).eta)<2.1 & mass > 60"),
    )


##ZMuMu: at least one HLT trigger match
goodZToMuMuAtLeast1HLT = cms.EDFilter(
    "ZGoldenSelectorAndFilter",
    zSelection,
    TrigTag = cms.InputTag("TriggerResults::HLT"),
    triggerEvent = cms.InputTag( "hltTriggerSummaryAOD::HLT" ),
    src = cms.InputTag("zmmCands"),
    condition =cms.string("atLeastOneMatched"),
    hltPath = cms.string("HLT_Mu9"),
    L3FilterName= cms.string("hltSingleMu9L3Filtered9"),
    maxDPtRel = cms.double( 1.0 ),
    maxDeltaR = cms.double( 0.2 ),
    filter = cms.bool(True) 
)




ewkZMuMuGoldenSequence = cms.Sequence(
    goodGlobalMuons 
# one may want to disregard the HLT filter
    # * dimuonsHLTFilter   
    * zmmCands 
    * dimuonsFilter 
# one may want to disregard the HLT matching
    # * goodZToMuMuAtLeast1HLT 
)


