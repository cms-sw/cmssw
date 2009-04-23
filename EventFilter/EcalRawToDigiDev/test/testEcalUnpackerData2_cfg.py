import FWCore.ParameterSet.Config as cms

process = cms.Process("ECALDQM")

process.load("EventFilter.EcalRawToDigiDev.EcalUnpackerMapping_cfi")
process.load("EventFilter.EcalRawToDigiDev.EcalUnpackerData_cfi")

process.load("Geometry.EcalMapping.EcalMapping_cfi")

process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

process.source = cms.Source("NewEventStreamFileReader",
     fileNames =
cms.untracked.vstring('file:/afs/cern.ch/user/f/franzoni/public/4nuno/ecal_local.00076678.0001.A.storageManager.00.0000.dat'))

process.ecalDataSequence = cms.Sequence(process.ecalEBunpacker)

process.p = cms.Path(process.ecalDataSequence)

process.ecalEBunpacker.silentMode =  True 
#process.ecalEBunpacker.silentMode = False 
