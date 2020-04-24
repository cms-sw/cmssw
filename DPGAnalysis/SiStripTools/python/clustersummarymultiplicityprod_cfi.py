import FWCore.ParameterSet.Config as cms

clustsummmultprod = cms.EDProducer("FromClusterSummaryMultiplicityProducer",
                                   clusterSummaryCollection = cms.InputTag("clusterSummaryProducer"),
                                   wantedSubDets = cms.VPSet(
    cms.PSet(detSelection = cms.uint32(0)   ,detLabel = cms.string("TK"),    subDetEnum = cms.int32(0)),
    cms.PSet(detSelection = cms.uint32(1)   ,detLabel = cms.string("TIB"),   subDetEnum = cms.int32(1)),
    cms.PSet(detSelection = cms.uint32(2)   ,detLabel = cms.string("TOB"),   subDetEnum = cms.int32(2)),
    cms.PSet(detSelection = cms.uint32(3)   ,detLabel = cms.string("TID"),   subDetEnum = cms.int32(3)),
    cms.PSet(detSelection = cms.uint32(4)   ,detLabel = cms.string("TEC"),   subDetEnum = cms.int32(4)),
    cms.PSet(detSelection = cms.uint32(1005),detLabel = cms.string("Pixel"), subDetEnum = cms.int32(5)),
    cms.PSet(detSelection = cms.uint32(1006),detLabel = cms.string("FPIX"),  subDetEnum = cms.int32(6)),
    cms.PSet(detSelection = cms.uint32(1007),detLabel = cms.string("BPIX"),  subDetEnum = cms.int32(7)),
    ),
                                                             
    varEnum = cms.int32(0)                             
    )
