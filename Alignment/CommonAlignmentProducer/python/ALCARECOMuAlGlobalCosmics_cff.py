# AlCaReco for muon alignment using global cosmic ray tracks
import FWCore.ParameterSet.Config as cms

# HLT
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOMuAlGlobalCosmicsHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
ALCARECOMuAlGlobalCosmicsHLT.andOr = True ## choose logical OR between Triggerbits
ALCARECOMuAlGlobalCosmicsHLT.HLTPaths = ["HLT_L1MuOpen", "HLT_L1Mu", "HLT_Mu3", "HLT_Mu5", "HLT_Mu7", "HLT_Mu9", "HLT_Mu11", "HLT_Mu13", "HLT_Mu15", "HLT_Mu15_L1Mu7", "HLT_L2Mu9", "HLT_IsoMu9", "HLT_IsoMu11", "HLT_IsoMu13", "HLT_IsoMu15"]

import Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi

ALCARECOMuAlGlobalCosmics = Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi.AlignmentMuonSelector.clone()
ALCARECOMuAlGlobalCosmics.src = cms.InputTag("GLBMuons")
ALCARECOMuAlGlobalCosmics.filter = cms.bool(True) # not strictly necessary, but provided for symmetry with MuAlStandAloneCosmics
ALCARECOMuAlGlobalCosmics.ptMin = cms.double(0.0)
ALCARECOMuAlGlobalCosmics.etaMin = cms.double(-100.0)
ALCARECOMuAlGlobalCosmics.etaMax = cms.double(100.0)

seqALCARECOMuAlGlobalCosmics = cms.Sequence(ALCARECOMuAlGlobalCosmicsHLT + ALCARECOMuAlGlobalCosmics)
