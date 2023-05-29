import FWCore.ParameterSet.Config as cms

# Specifications for Correlator Layer 1 output mapping onto links
barrelConfig_ = cms.PSet(
    partition = cms.string("Barrel"),
    nLinksPuppi = cms.uint32(6),
    nPuppiPerRegion = cms.uint32(18),
    nOutputFramesPerBX = cms.uint32(9),
)

barrelPhiConfigs = [
    barrelConfig_.clone(
        outputRegions = cms.vuint32(*[3*ip+9*ie+i for ie in range(6) for i in range(3) ]),
        outputBoard = cms.int32(ip),
    ) for ip in range(3)
]

hgcalConfig_ = cms.PSet(
    partition = cms.string("HGCal"),
    nLinksPuppi = cms.uint32(3),
    nPuppiPerRegion = cms.uint32(18),
    nOutputFramesPerBX = cms.uint32(9),
    outputRegions = cms.vuint32(*[54 + i+9 for i in range(9)]),
    outputBoard = cms.int32(3),
)

hgcalPosConfig = hgcalConfig_.clone(
    outputBoard = 4
)
hgcalNegConfig = hgcalConfig_.clone(
    outputRegions = [54 + i for i in range(9)]
)

hgcalNoTKConfig = cms.PSet(
    partition = cms.string("HGCalNoTk"),
    nLinksPuppi = cms.uint32(4),
    nPuppiPerRegion = cms.uint32(12),
    nOutputFramesPerBX = cms.uint32(9),
    outputRegions = cms.vuint32(*range(72,72+18)),
    outputBoard = cms.int32(5),
)

hfConfig_ = cms.PSet(
    partition = cms.string("HF"),
    nLinksPuppi = cms.uint32(3),
    nPuppiPerRegion = cms.uint32(18),
    nOutputFramesPerBX = cms.uint32(9),
)
hfConfigs = [
    hfConfig_.clone(
        outputRegions = cms.vuint32(*[90+9*ie+i for i in range(9)]),
        outputBoard = cms.int32(6 + ie),
    ) for ie in range(2)
]

linkConfigs = cms.VPSet(*barrelPhiConfigs, hgcalPosConfig, hgcalNegConfig, hgcalNoTKConfig, *hfConfigs)

l1tDeregionizerProducer = cms.EDProducer("DeregionizerProducer",
                           RegionalPuppiCands  = cms.InputTag("l1tLayer1","PuppiRegional"),
                           nPuppiFinalBuffer   = cms.uint32(128),
                           nPuppiPerClk        = cms.uint32(6),
                           nPuppiFirstBuffers  = cms.uint32(12),
                           nPuppiSecondBuffers = cms.uint32(32),
                           nPuppiThirdBuffers  = cms.uint32(64),
                           nInputFramesPerBX   = cms.uint32(9),
                           linkConfigs         = linkConfigs,
                         )

l1tDeregionizerProducerExtended = l1tDeregionizerProducer.clone(RegionalPuppiCands  = cms.InputTag("l1tLayer1Extended","PuppiRegional"))
