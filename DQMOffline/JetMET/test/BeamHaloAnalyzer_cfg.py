import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
process.load("Configuration/StandardSequences/RawToDigi_Data_cff")

process.load("RecoMET/METProducers/BeamHaloSummary_cfi")
process.load("RecoMET/METProducers/CSCHaloData_cfi")
process.load("RecoMET/METProducers/EcalHaloData_cfi")
process.load("RecoMET/METProducers/HcalHaloData_cfi")
process.load("RecoMET/METProducers/GlobalHaloData_cfi")
#process.GlobalTag.globaltag = 'STARTUP31X_V1::All'
process.GlobalTag.globaltag ='STARTUP3X_V14::All'
process.DQMStore = cms.Service("DQMStore")

process.load("Configuration/StandardSequences/ReconstructionCosmics_cff")

process.load("RecoMuon/Configuration/RecoMuon_cff")

process.load("DQMOffline/JetMET/BeamHaloAnalyzer_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.load("DQMServices.Components.DQMStoreStats_cfi")

#process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_4_1/RelValBeamHalo/GEN-SIM-RECO/STARTUP3X_V14-v1/0005/DA97EC42-21EE-DE11-A0D1-003048D3750A.root',
                '/store/relval/CMSSW_3_4_1/RelValBeamHalo/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/6684C797-C9ED-DE11-BC19-0030487C6090.root'
    )
                            )
process.p = cms.Path(process.BeamHaloId*process.AnalyzeBeamHalo*process.dqmStoreStats)

#### If cosmic muons are not by default in the event, then you should run this sequence
#process.p = cms.Path(process.ctfWithMaterialTracksP5LHCNavigation*process.muonRecoLHC*process.CSCHaloData*process.EcalHaloData*process.HcalHaloData*process.GlobalHaloData*process.AnalyzeBeamHalo)
