# AlCaReco for muon alignment using straight (zero-field) cosmic ray tracks
import FWCore.ParameterSet.Config as cms

# HLT
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOMuAlZeroFieldGlobalCosmicsHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
ALCARECOMuAlZeroFieldGlobalCosmicsHLT.andOr = True ## choose logical OR between Triggerbits
ALCARECOMuAlZeroFieldGlobalCosmicsHLT.HLTPaths = ['HLT_TrackerCosmics', 'HLT_TrackerCosmics_CoTF', 'HLT_IsoMu11', 'HLT_Mu15_L1Mu7']

import Alignment.CommonAlignmentProducer.ZeroFieldGlobalMuonBuilder_cfi

ALCARECOMuAlZeroFieldGlobalCosmics = Alignment.CommonAlignmentProducer.ZeroFieldGlobalMuonBuilder_cfi.ZeroFieldGlobalMuonBuilder.clone()

seqALCARECOMuAlZeroFieldGlobalCosmics = cms.Sequence(ALCARECOMuAlZeroFieldGlobalCosmicsHLT+ALCARECOMuAlZeroFieldGlobalCosmics)
