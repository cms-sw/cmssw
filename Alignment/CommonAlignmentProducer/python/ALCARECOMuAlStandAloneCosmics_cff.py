# AlCaReco for muon alignment using stand-alone cosmic ray tracks
import FWCore.ParameterSet.Config as cms

# HLT
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOMuAlStandAloneCosmicsHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
ALCARECOMuAlStandAloneCosmicsHLT.andOr = True ## choose logical OR between Triggerbits
ALCARECOMuAlStandAloneCosmicsHLT.HLTPaths = ["HLT_L1MuOpen", "HLT_L1Mu", "HLT_Mu3", "HLT_Mu5", "HLT_Mu7", "HLT_Mu9", "HLT_Mu11", "HLT_Mu13", "HLT_Mu15", "HLT_Mu15_L1Mu7", "HLT_L2Mu9", "HLT_IsoMu9", "HLT_IsoMu11", "HLT_IsoMu13", "HLT_IsoMu15"]

import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi

ALCARECOMuAlStandAloneCosmics = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
ALCARECOMuAlStandAloneCosmics.src = cms.InputTag("cosmicMuons")
ALCARECOMuAlStandAloneCosmics.filter = cms.bool(True)
ALCARECOMuAlStandAloneCosmics.ptMin = cms.double(0.0)
ALCARECOMuAlStandAloneCosmics.etaMin = cms.double(-100.0)
ALCARECOMuAlStandAloneCosmics.etaMax = cms.double(100.0)

# Turn off trigger requirement for first CRUZET-4/CRAFT test
# seqALCARECOMuAlStandAloneCosmics = cms.Sequence(ALCARECOMuAlStandAloneCosmicsHLT + ALCARECOMuAlStandAloneCosmics)
seqALCARECOMuAlStandAloneCosmics = cms.Sequence(ALCARECOMuAlStandAloneCosmics)

