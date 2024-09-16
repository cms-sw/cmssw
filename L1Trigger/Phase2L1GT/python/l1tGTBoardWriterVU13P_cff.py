import FWCore.ParameterSet.Config as cms
from L1Trigger.Phase2L1GT.l1tGTBoardWriter_cff import BoardDataInput as BoardDataInputVU13P
from L1Trigger.Phase2L1GT.l1tGTBoardWriter_cff import BoardDataOutputObjects as BoardDataOutputObjectsVU13P
from L1Trigger.Phase2L1GT.l1tGTBoardWriter_cff import AlgoBitBoardData as AlgoBitBoardDataVU13P

BoardDataInputVU13P.InputChannels = cms.untracked.PSet(
    # SLR 0
    GTT_1 = cms.untracked.vuint32(range(0, 6)),
    GTT_2 = cms.untracked.vuint32(range(6, 12)),
    GTT_3 = cms.untracked.vuint32(range(112, 118)),
    GTT_4 = cms.untracked.vuint32(range(118, 124)),

    # SLR 1
    GCT_1 = cms.untracked.vuint32(range(24, 30)),

    # SLR 2
    CL2_1 = cms.untracked.vuint32(range(32, 38)),
    CL2_2 = cms.untracked.vuint32(range(38, 44)),
    CL2_3 = cms.untracked.vuint32(range(80, 86)),
    
    # SLR 3
    GMT_1 = cms.untracked.vuint32(48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 68, 69, 70, 71, 72, 73)
)

BoardDataOutputObjectsVU13P.OutputChannels = cms.untracked.PSet(
    GTTPromptJets = cms.untracked.vuint32(range(2, 6)),
    GTTDisplacedJets = cms.untracked.vuint32(range(6, 10)),
    GTTPromptHtSum = cms.untracked.vuint32(range(10, 11)),
    GTTDisplacedHtSum = cms.untracked.vuint32(range(11, 12)),
    GTTEtSum = cms.untracked.vuint32(range(12, 13)),
    GTTHadronicTaus = cms.untracked.vuint32(range(13, 16)),
    GCTNonIsoEg = cms.untracked.vuint32(range(26, 28)),
    GCTIsoEg = cms.untracked.vuint32(range(28, 30)),
    GCTJets = cms.untracked.vuint32(range(30, 32)),
    CL2JetsSC4 = cms.untracked.vuint32(range(32, 36)),
    CL2JetsSC8 = cms.untracked.vuint32(range(36, 40)),
    CL2Taus = cms.untracked.vuint32(range(40, 43)),
    CL2HtSum = cms.untracked.vuint32(range(43, 44)),
    CL2EtSum = cms.untracked.vuint32(range(44, 45)),
    GMTSaPromptMuons = cms.untracked.vuint32(range(68, 70)),
    GMTSaDisplacedMuons = cms.untracked.vuint32(range(70, 72)),
    GMTTkMuons = cms.untracked.vuint32(range(72, 75)),
    GMTTopo = cms.untracked.vuint32(range(75, 77)),
    CL2Electrons = cms.untracked.vuint32(range(80, 83)),
    CL2Photons = cms.untracked.vuint32(range(83, 86)),
    GCTTaus = cms.untracked.vuint32(range(96, 98)),
    GCTHtSum = cms.untracked.vuint32(range(98, 99)),
    GCTEtSum = cms.untracked.vuint32(range(99, 100)),
    GTTPhiCandidates = cms.untracked.vuint32(range(112, 115)),
    GTTRhoCandidates = cms.untracked.vuint32(range(115, 118)),
    GTTBsCandidates = cms.untracked.vuint32(range(118, 121)),
    GTTPrimaryVert = cms.untracked.vuint32(range(121, 123))
)

AlgoBitBoardDataVU13P.channels = cms.untracked.vuint32(46, 47)
