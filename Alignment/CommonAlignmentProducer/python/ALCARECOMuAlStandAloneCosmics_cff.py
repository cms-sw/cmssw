# AlCaReco for muon alignment using straight (zero-field) cosmic ray tracks
import FWCore.ParameterSet.Config as cms

# HLT
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOMuAlStandAloneCosmicsHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
ALCARECOMuAlStandAloneCosmicsHLT.andOr = True ## choose logical OR between Triggerbits
ALCARECOMuAlStandAloneCosmicsHLT.HLTPaths = ['HLT_TrackerCosmics', 'HLT_TrackerCosmics_CoTF', 'HLT_IsoMu11', 'HLT_Mu15_L1Mu7']

import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi

ALCARECOMuAlStandAloneCosmics = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
ALCARECOMuAlStandAloneCosmics.src = cms.InputTag("cosmicMuons")
ALCARECOMuAlStandAloneCosmics.filter = cms.bool(True)
ALCARECOMuAlStandAloneCosmics.ptMin = cms.double(0.0)
ALCARECOMuAlStandAloneCosmics.etaMin = cms.double(-100.0)
ALCARECOMuAlStandAloneCosmics.etaMax = cms.double(100.0)

seqALCARECOMuAlStandAloneCosmics = cms.Sequence(ALCARECOMuAlStandAloneCosmicsHLT+ALCARECOMuAlStandAloneCosmics)
