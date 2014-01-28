import FWCore.ParameterSet.Config as cms

process = cms.Process("RCTofflineTEST")

#process.load("DQMServices.Core.DQM_cfg")
process.DQMStore = cms.Service("DQMStore")

process.load("DQM/L1TMonitor/L1TRCToffline_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#if you use the dbs
'/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/774/2260AF5F-356E-DD11-99F3-000423D6C8E6.root'
#if you use castor
#'rfio:/castor/cern.ch/cms/store/data/BeamCommissioning08/Cosmics/RAW/v1/000/063/135/86E93798-5B85-DD11-BE81-000423D9870C.root'
))


