import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
# AlCaReco for muon alignment using straight (zero-field) cosmic ray tracks
ALCARECOMuAlZeroFieldGlobalCosmicsHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
import Alignment.CommonAlignmentProducer.ZeroFieldGlobalMuonBuilder_cfi
ALCARECOMuAlZeroFieldGlobalCosmics = Alignment.CommonAlignmentProducer.ZeroFieldGlobalMuonBuilder_cfi.ZeroFieldGlobalMuonBuilder.clone()
seqALCARECOMuAlZeroFieldGlobalCosmics = cms.Sequence(ALCARECOMuAlZeroFieldGlobalCosmicsHLT+ALCARECOMuAlZeroFieldGlobalCosmics)
ALCARECOMuAlZeroFieldGlobalCosmicsHLT.andOr = True ## choose logical OR between Triggerbits

ALCARECOMuAlZeroFieldGlobalCosmicsHLT.HLTPaths = ['CandHLTTrackerCosmics', 'CandHLTTrackerCosmicsCoTF', 'HLT_IsoMu11', 'HLT_Mu15_L1Mu7']

