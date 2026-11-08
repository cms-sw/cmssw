import FWCore.ParameterSet.Config as cms

l1tStubsGmt = cms.EDProducer("Phase2L1TGMTStubProducer",
    Barrel = cms.PSet(
        coarseEta_1 = cms.vint32(0, 23, 41),
        coarseEta_2 = cms.vint32(0, 20, 36),
        coarseEta_3 = cms.vint32(0, 17, 31),
        coarseEta_4 = cms.vint32(0, 14, 27),
        etaLSB = cms.double(0.024586688),
        eta_1 = cms.vint32(
            -46, -45, -43, -41, -39,
            -37, -35, -30, -28, -26,
            -23, -20, -18, -15, -9,
            -6, -3, -1, 1, 3,
            6, 9, 15, 18, 20,
            23, 26, 28, 30, 35,
            37, 39, 41, 43, 45,
            1503
        ),
        eta_2 = cms.vint32(
            -41, -39, -38, -36, -34,
            -32, -30, -26, -24, -22,
            -20, -18, -15, -13, -8,
            -5, -3, -1, 1, 3,
            5, 8, 13, 15, 18,
            20, 22, 24, 26, 30,
            32, 34, 36, 38, 39,
            1334
        ),
        eta_3 = cms.vint32(
            -35, -34, -32, -31, -29,
            -27, -26, -22, -20, -19,
            -17, -15, -13, -11, -6,
            -4, -2, -1, 1, 2,
            4, 6, 11, 13, 15,
            17, 19, 20, 22, 26,
            27, 29, 31, 32, 34,
            1148
        ),
        maxBX = cms.int32(0),
        minBX = cms.int32(0),
        minPhiQuality = cms.int32(0),
        minThetaQuality = cms.int32(0),
        phiBDivider = cms.int32(1),
        phiLSB = cms.double(2.396845e-05),
        phiOffset = cms.vint32(0, 0, 0, 0),
        verbose = cms.int32(0)
    ),
    Endcap = cms.PSet(
        coord1LSB = cms.double(0.00613281248),
        coord2LSB = cms.double(0.00613281248),
        eta1LSB = cms.double(0.024586688),
        eta2LSB = cms.double(0.024586688),
        etaMatch = cms.double(0.1),
        maxBX = cms.int32(0),
        minBX = cms.int32(0),
        phiMatch = cms.double(0.05),
        verbose = cms.uint32(0)
    ),
    srcCSC = cms.InputTag("simCscTriggerPrimitiveDigis"),
    srcDT = cms.InputTag("dtTriggerPhase2PrimitiveDigis"),
    srcDTPairs = cms.InputTag("dtTriggerPhase2PrimitivePairDigis"),
    srcDTTheta = cms.InputTag("simDtTriggerPrimitiveDigis"),
    srcRPC = cms.InputTag("simMuonRPCDigis"),
    verbose = cms.int32(0)
)
