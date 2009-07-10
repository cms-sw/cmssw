import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTECALUNPACKERDATA")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
 fileNames = 
  cms.untracked.vstring('/store/relval/CMSSW_3_1_0/RelValQCD_Pt_3000_3500/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/1A1C4478-5866-DE11-A907-001D09F27067.root')
#cms.untracked.vstring('file:/tmp/nalmeida/1A1C4478-5866-DE11-A907-001D09F27067.root')


)

process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")


process.load("EventFilter.EcalRawToDigi.EcalUnpackerMapping_cfi")
process.load("EventFilter.EcalRawToDigi.EcalUnpackerData_cfi")

process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")


# this is for MC
process.ecalEBunpacker.InputLabel = cms.string('rawDataCollector')
# this is for real data
# process.ecalEBunpacker.InputLabel = cms.string('source')

# print out debug info ( put silent mode = False)
#process.ecalEBunpacker.silentMode =  True 
process.ecalEBunpacker.silentMode = False 



process.ecalDataSequence = cms.Sequence(process.ecalEBunpacker)

process.p = cms.Path(process.ecalDataSequence)




