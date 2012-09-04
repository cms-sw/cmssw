import FWCore.ParameterSet.Config as cms
from DQM.EcalCommon.dqmpset import *
from DQM.EcalBarrelMonitorClient.IntegrityClient_cfi import integrityClient

integrity = dqmpset(integrityClient)

ecalCondDBWriter = cms.EDAnalyzer("EcalCondDBWriter",
    DBName = cms.untracked.string(""),
    hostName = cms.untracked.string(""),
    hostPort = cms.untracked.int32(0),
    userName = cms.untracked.string(""),
    password = cms.untracked.string(""),
    tagName = cms.untracked.string(""),
    location = cms.untracked.string(""),
    runType = cms.untracked.string(""),
    inputRootFiles = cms.untracked.vstring(),
    MESetParams = cms.untracked.PSet(
        Integrity = cms.untracked.PSet(
            Quality = integrity.MEs.Quality
        ),
        Cosmic = cms.untracked.PSet(),
        Laser = cms.untracked.PSet(),
        Pedestal = cms.untracked.PSet(),
        Presample = cms.untracked.PSet(),
        TestPulse = cms.untracked.PSet(),
        BeamCalo = cms.untracked.PSet(),
        BeamHodo = cms.untracked.PSet(),
        TriggerPrimitives = cms.untracked.PSet(),
        Cluster = cms.untracked.PSet(),
        Timing = cms.untracked.PSet(),
        Led = cms.untracked.PSet(),
        RawData = cms.untracked.PSet(),
        Occupancy = cms.untracked.PSet()
    )
)
    
