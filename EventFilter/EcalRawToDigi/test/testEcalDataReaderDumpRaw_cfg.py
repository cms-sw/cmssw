# Configuration file to test the EcalDataReader module. It
# dumps raw data unsing the EcalDataReader module dump 
# functionality.
#
# Original author: Ph. Gras CEA/Saclay

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.load("EventFilter.EcalRawToDigi.EcalDataReader_cfi")
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")
# trivial conditions for ECAL Channel Status == 0
process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")

process.source = cms.Source("PoolSource",
 fileNames = 
 cms.untracked.vstring('file:/tmp/pgras/FECE8EF7-6A8F-E211-9748-002618943969.root'),
# cms.untracked.vstring('/store/data/Run2012D/DoubleElectron/RAW-RECO/ZElectron-22Jan2013-v1/10000/FECE8EF7-6A8F-E211-9748-002618943969.root'),
 inputCommands=cms.untracked.vstring(
                  'drop *',
                  'keep FEDRawDataCollection_*_*_*'
                  )
)

process.output = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string('digi.root')
)

process.path = cms.Path(process.ecalDumpRaw)

