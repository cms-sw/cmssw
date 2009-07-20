import FWCore.ParameterSet.Config as cms

APVPhases = cms.EDProducer('ConfigurableAPVCyclePhaseProducer',
                           defaultPartitionNames = cms.vstring("TI_13-JUN-2009_1",
                                                               "TO_30-JUN-2009_1",
                                                               "TP_09-JUN-2009_1",
                                                               "TM_09-JUN-2009_1"
                                                               ),
                           defaultPhases = cms.vint32(-1,-1,-1,-1),
                           runPhases = cms.VPSet(
                               cms.PSet( runNumber = cms.int32(100967),phases = cms.untracked.vint32(30),partitions = cms.untracked.vstring("TM_09-JUN-2009_1")),
                               cms.PSet( runNumber = cms.int32(100995),phases = cms.untracked.vint32(30),partitions = cms.untracked.vstring("TM_09-JUN-2009_1")),
                               cms.PSet( runNumber = cms.int32(101012),phases = cms.untracked.vint32(30),partitions = cms.untracked.vstring("TM_09-JUN-2009_1")),
                               cms.PSet( runNumber = cms.int32(101018),phases = cms.untracked.vint32(30),partitions = cms.untracked.vstring("TM_09-JUN-2009_1")),
                               cms.PSet( runNumber = cms.int32(101043),phases = cms.untracked.vint32(30),partitions = cms.untracked.vstring("TM_09-JUN-2009_1")),
                               cms.PSet( runNumber = cms.int32(101045),phases = cms.untracked.vint32(30),partitions = cms.untracked.vstring("TM_09-JUN-2009_1")),
                               cms.PSet( runNumber = cms.int32(102130),phases = cms.untracked.vint32(30),partitions = cms.untracked.vstring("TM_09-JUN-2009_1")),
                               cms.PSet( runNumber = cms.int32(102169),phases = cms.untracked.vint32(30),partitions = cms.untracked.vstring("TM_09-JUN-2009_1")),
                               cms.PSet( runNumber = cms.int32(105612),phases = cms.untracked.vint32(-1,-1,-1,-1)),
                               cms.PSet( runNumber = cms.int32(105755),phases = cms.untracked.vint32(30,30,30,30)),
                               cms.PSet( runNumber = cms.int32(105765),phases = cms.untracked.vint32(30,30,30,30)),
                               cms.PSet( runNumber = cms.int32(105820),phases = cms.untracked.vint32(30,30,30,30)),
                               cms.PSet( runNumber = cms.int32(106019),phases = cms.untracked.vint32(30,30,30,30))
                                                 )
)
