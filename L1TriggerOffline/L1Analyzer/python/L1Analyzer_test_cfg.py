import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.load("FWCore.MessageService.MessageLogger_cfi")

# L1 GT EventSetup
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")
#process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup2_v2_Unprescaled_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup2_v4_L1T_Scales_20080926_startup_Imp0_Unprescaled_cff")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000) )

process.source = cms.Source("PoolSource",
                           fileNames = cms.untracked.vstring(

##run 66757
  '/store/data/Commissioning08/Calo/RECO/v1/000/066/757/006C110E-B59E-DD11-B84A-001617E30CA4.root',
  '/store/data/Commissioning08/Calo/RECO/v1/000/066/757/40B597E2-B29E-DD11-AC62-001617E30D38.root',
  '/store/data/Commissioning08/Calo/RECO/v1/000/066/757/ACA165E4-B29E-DD11-82DA-001617C3B5E4.root'

)
)

# L1 EventSetup
process.load("L1Trigger.Configuration.L1DummyConfig_cff")

# Extract the L1GTriggerReadoutRecord
process.load("L1TriggerOffline.L1Analyzer.TriggerOperation_cfi")

process.load("L1TriggerOffline.L1Analyzer.L1CenJetRecoAnalysis_cff")
process.load("L1TriggerOffline.L1Analyzer.L1TauJetRecoAnalysis_cff")

process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")
process.load("L1Trigger.Skimmer.l1Filter_cfi")
process.l1Filter.algorithms = cms.vstring('L1_SingleEG1','L1_SingleEG5','L1_SingleEG8','L1_SingleEG10','L1_SingleEG12','L1_SingleEG15','L1_SingleEG20')

process.test = cms.Path(process.demo+process.l1Filter+process.L1TauJetRecoAnalysis+process.L1CenJetRecoAnalysis)

