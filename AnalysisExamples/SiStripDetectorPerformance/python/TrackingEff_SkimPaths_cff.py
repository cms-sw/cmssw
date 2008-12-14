import FWCore.ParameterSet.Config as cms

from AnalysisExamples.SiStripDetectorPerformance.TrackingEff_Sequences_cff import *
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
TrackingEffHLTFilter = copy.deepcopy(hltHighLevel)
tagmuonFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("TagMuons"),
    minNumber = cms.uint32(1)
)

probemuonFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("goodStandAloneMuonTracks"),
    minNumber = cms.uint32(2)
)

TrackingEffPath = cms.Path(TrackingEffHLTFilter+muonRecoForTrackingEff+tagmuonFilter+probemuonFilter)
TrackingEffHLTFilter.HLTPaths = ['HLT1MuonNonIso', 'CandHLT1MuonPrescalePt5', 'CandHLT1MuonPrescalePt7x7', 'CandHLT1MuonPrescalePt7x10']

