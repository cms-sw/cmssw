# AlCaReco for muon alignment using stand-alone cosmic ray tracks
import FWCore.ParameterSet.Config as cms

# HLT
#import HLTrigger.HLTfilters.hltHighLevel_cfi
#ALCARECOMuAlStandAloneCosmicsHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
#    andOr = True, ## choose logical OR between Triggerbits
#    HLTPaths = ["HLT_L1MuOpen", "HLT_L1Mu", "HLT_Mu3", "HLT_Mu5", "HLT_Mu7", "HLT_Mu9",
#                "HLT_Mu11", "HLT_Mu13", "HLT_Mu15", "HLT_L2Mu9", "HLT_IsoMu9", "HLT_IsoMu11",
#                "HLT_IsoMu13", "HLT_IsoMu15"],
#    throw = False # tolerate triggers stated above, but not available
#    )
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi

ALCARECOMuAlStandAloneCosmics = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone(
    src = "cosmicMuons", #cms.InputTag("cosmicMuons")
    filter = True,
    ptMin = 0.0,
    etaMin = -100.0,
    etaMax = 100.0
    )

#seqALCARECOMuAlStandAloneCosmics = cms.Sequence(ALCARECOMuAlStandAloneCosmicsHLT + ALCARECOMuAlStandAloneCosmics)
seqALCARECOMuAlStandAloneCosmics = cms.Sequence(ALCARECOMuAlStandAloneCosmics)

