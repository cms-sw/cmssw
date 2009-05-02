import FWCore.ParameterSet.Config as cms

process = cms.Process("ECALDQM")

process.load("EventFilter.EcalRawToDigiDev.EcalUnpackerMapping_cfi")
process.load("EventFilter.EcalRawToDigiDev.EcalUnpackerData_cfi")

process.load("Geometry.EcalMapping.EcalMapping_cfi")

process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

process.source = cms.Source("NewEventStreamFileReader",
     fileNames =
cms.untracked.vstring(
'rfio:/castor/cern.ch/user/f/franzoni/forNuno/ecal_local.00081763.0001.A.storageManager.00.0000.dat'
,'rfio:/castor/cern.ch/user/f/franzoni/forNuno/ecal_local.00081757.0001.A.storageManager.00.0000.dat'
,'rfio:/castor/cern.ch/user/f/franzoni/forNuno/ecal_local.00081750.0001.A.storageManager.00.0000.dat'
,'rfio:/castor/cern.ch/user/f/franzoni/forNuno/ecal_local.00081751.0001.A.storageManager.00.0000.dat'
,'rfio:/castor/cern.ch/user/f/franzoni/forNuno/ecal_local.00081753.0001.A.storageManager.00.0000.dat'

))
#ecal_local.00081749.0001.A.storageManager.00.0000.dat
#ecal_local.00081750.0001.A.storageManager.00.0000.dat
#ecal_local.00081751.0001.A.storageManager.00.0000.dat
#ecal_local.00081753.0001.A.storageManager.00.0000.dat
#ecal_local.00081757.0001.A.storageManager.00.0000.dat
#ecal_local.00081763.0001.A.storageManager.00.0000.dat

process.ecalDataSequence = cms.Sequence(process.ecalEBunpacker)

process.p = cms.Path(process.ecalDataSequence)

process.ecalEBunpacker.silentMode =  True 
#process.ecalEBunpacker.silentMode = False 



