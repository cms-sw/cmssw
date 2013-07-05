import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTECALUNPACKERDATA")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
 fileNames = 
#cms.untracked.vstring('file:/tmp/nalmeida/1A1C4478-5866-DE11-A907-001D09F27067.root')
 cms.untracked.vstring('/store/relval/CMSSW_3_8_0/RelValProdTTbar/GEN-SIM-RAW/MC_38Y_V7-v1/0004/306D1971-0C95-DF11-942B-002618943984.root')

)

process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")


process.load("EventFilter.EcalRawToDigi.EcalUnpackerMapping_cfi")
process.load("EventFilter.EcalRawToDigi.EcalUnpackerData_cfi")

# trivial conditions for ECAL Channel Status == 0
process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")

# conditions of real runs:
#process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
# global tag for ECAL Channel Status DB (be sure to use one which correspond to your version of CMSSW)
#process.GlobalTag.globaltag = 'GR_R_35X_V8A::All'


# this is for MC
process.ecalEBunpacker.InputLabel = cms.InputTag('rawDataCollector')
# this is for real data
# process.ecalEBunpacker.InputLabel = cms.InputTag('rawDataCollector')

# print out debug info ( put silent mode = False)
#process.ecalEBunpacker.silentMode =  True 
process.ecalEBunpacker.silentMode = False 

process.ecalEBunpacker.syncCheck = True 

process.ecalDataSequence = cms.Sequence(process.ecalEBunpacker)

process.p = cms.Path(process.ecalDataSequence)




