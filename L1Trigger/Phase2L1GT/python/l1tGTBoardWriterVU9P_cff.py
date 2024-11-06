import FWCore.ParameterSet.Config as cms
from L1Trigger.Phase2L1GT.l1tGTBoardWriter_cff import BoardDataInput as BoardDataInputVU9P
from L1Trigger.Phase2L1GT.l1tGTBoardWriter_cff import BoardDataOutputObjects as BoardDataOutputObjectsVU9P
from L1Trigger.Phase2L1GT.l1tGTBoardWriter_cff import AlgoBitBoardData as AlgoBitBoardDataVU9P


BoardDataInputVU9P.InputChannels = cms.untracked.PSet(
    # SLR 0
    GTT_1 = cms.untracked.vuint32(range(0, 6)),
    GTT_2 = cms.untracked.vuint32(range(6, 12)),
    GTT_3 = cms.untracked.vuint32(range(104, 110)),
    GTT_4 = cms.untracked.vuint32(range(110, 116)),
    
    # SLR 1
    CL2_1 = cms.untracked.vuint32(range(28, 34)),
    CL2_2 = cms.untracked.vuint32(range(34, 40)),
    CL2_3 = cms.untracked.vuint32(range(80, 86)),

    # SLR 2
    GCT_1 = cms.untracked.vuint32(range(54, 60)),
    GMT_1 = cms.untracked.vuint32(range(60, 78))
)

BoardDataOutputObjectsVU9P.OutputChannels = cms.untracked.PSet(
    GTTPromptJets = cms.untracked.vuint32(range(2, 6)),
    GTTDisplacedJets = cms.untracked.vuint32(range(6, 10)),
    GTTPromptHtSum = cms.untracked.vuint32(range(10, 11)),
    GTTDisplacedHtSum = cms.untracked.vuint32(range(11, 12)),
    GTTEtSum = cms.untracked.vuint32(range(12, 13)),
    GTTHadronicTaus = cms.untracked.vuint32(range(13, 16)),
    CL2JetsSC4 = cms.untracked.vuint32(range(24, 28)),
    CL2JetsSC8 = cms.untracked.vuint32(range(28, 32)),
    CL2Taus = cms.untracked.vuint32(range(34, 37)),
    CL2HtSum = cms.untracked.vuint32(range(37, 38)),
    CL2EtSum = cms.untracked.vuint32(range(38, 39)),
    GCTNonIsoEg = cms.untracked.vuint32(range(48, 50)),
    GCTIsoEg = cms.untracked.vuint32(range(50, 52)),
    GCTJets = cms.untracked.vuint32(range(52, 54)),
    GCTTaus = cms.untracked.vuint32(range(54, 56)),
    GCTHtSum = cms.untracked.vuint32(range(56, 57)),
    GCTEtSum = cms.untracked.vuint32(range(57, 58)),
    GMTSaPromptMuons = cms.untracked.vuint32(range(60, 62)),
    GMTSaDisplacedMuons = cms.untracked.vuint32(range(62, 64)),
    GMTTkMuons = cms.untracked.vuint32(range(64, 67)),
    GMTTopo = cms.untracked.vuint32(range(67, 69)),
    CL2Electrons = cms.untracked.vuint32(range(80, 83)),
    CL2Photons = cms.untracked.vuint32(range(83, 86)),
    GTTPhiCandidates = cms.untracked.vuint32(range(104, 107)),
    GTTRhoCandidates = cms.untracked.vuint32(range(107, 110)),
    GTTBsCandidates = cms.untracked.vuint32(range(110, 113)),
    GTTPrimaryVert = cms.untracked.vuint32(range(113, 115))
)

AlgoBitBoardDataVU9P.channels = cms.untracked.vuint32(32, 33)
