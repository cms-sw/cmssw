# AlCaReco for muon alignment using straight (zero-field) cosmic ray tracks
import FWCore.ParameterSet.Config as cms

# HLT
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOMuAlGlobalCosmicsHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
ALCARECOMuAlGlobalCosmicsHLT.andOr = True ## choose logical OR between Triggerbits
ALCARECOMuAlGlobalCosmicsHLT.HLTPaths = ['HLT_TrackerCosmics', 'HLT_TrackerCosmics_CoTF', 'HLT_IsoMu11', 'HLT_Mu15_L1Mu7']

import Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi

ALCARECOMuAlGlobalCosmics = Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi.AlignmentMuonSelector.clone()
ALCARECOMuAlGlobalCosmics.src = cms.InputTag("GLBMuons")
ALCARECOMuAlGlobalCosmics.ptMin = cms.double(0.0)
ALCARECOMuAlGlobalCosmics.etaMin = cms.double(-100.0)
ALCARECOMuAlGlobalCosmics.etaMax = cms.double(100.0)

seqALCARECOMuAlGlobalCosmics = cms.Sequence(ALCARECOMuAlGlobalCosmicsHLT+ALCARECOMuAlGlobalCosmics)
