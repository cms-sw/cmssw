import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOMuAlZeroFieldGlobalCosmicsHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
ALCARECOMuAlZeroFieldGlobalCosmicsHLT.andOr = True
ALCARECOMuAlZeroFieldGlobalCosmicsHLT.HLTPaths = ["CandHLTTrackerCosmics", "CandHLTTrackerCosmicsCoTF", "HLT_IsoMu11", "HLT_Mu15_L1Mu7"]

import Alignment.CommonAlignmentProducer.ZeroFieldGlobalMuonBuilder_cfi
ALCARECOMuAlZeroFieldGlobalCosmics = Alignment.CommonAlignmentProducer.ZeroFieldGlobalMuonBuilder_cfi.ZeroFieldGlobalMuonBuilder.clone()

seqALCARECOMuAlZeroFieldGlobalCosmics = cms.Sequence(ALCARECOMuAlZeroFieldGlobalCosmicsHLT + ALCARECOMuAlZeroFieldGlobalCosmics)
