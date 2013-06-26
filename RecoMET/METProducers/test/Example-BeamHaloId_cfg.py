import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("Configuration/StandardSequences/Geometry_cff")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
process.load("Configuration/StandardSequences/RawToDigi_Data_cff")
process.load("RecoMET/METProducers/BeamHaloSummary_cfi")
process.load("RecoMET/METProducers/CSCHaloData_cfi")
process.load("RecoMET/METProducers/EcalHaloData_cfi")
process.load("RecoMET/METProducers/HcalHaloData_cfi")
process.load("RecoMET/METProducers/GlobalHaloData_cfi")
process.load("RecoMET/Configuration/RecoMET_BeamHaloId_cff")
process.DQMStore = cms.Service("DQMStore")
process.load("DQMOffline/JetMET/BeamHaloAnalyzer_cfi")
process.AnalyzeBeamHalo.OutputFile  = cms.string("BeamHaloData.root")
process.GlobalTag.globaltag ='STARTUP3X_V11::All'
#process.GlobalTag.globaltag ='STARTUP31X_V7::All'
#process.GlobalTag.globaltag = 'STARTUP3XY_V9::All'

process.load("Configuration/StandardSequences/ReconstructionCosmics_cff")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
                            debugFlag = cms.untracked.bool(True),
                            debugVebosity = cms.untracked.uint32(10),
                            fileNames = cms.untracked.vstring(

    # 340pre2 RelVal
    #'/store/relval/CMSSW_3_4_0_pre2/RelValBeamHalo/GEN-SIM-RECO/STARTUP3XY_V9-v1/0003/94C6A8AC-C0BD-DE11-9B6D-003048678FD6.root',
    #'/store/relval/CMSSW_3_4_0_pre2/RelValBeamHalo/GEN-SIM-RECO/STARTUP3XY_V9-v1/0003/7A81B838-89BD-DE11-9A1A-0018F3D09678.root'
        

    # 340pre5 RelVal
    '/store/relval/CMSSW_3_4_0_pre5/RelValBeamHalo/GEN-SIM-RECO/STARTUP3X_V11-v1/0002/4E694747-F9CB-DE11-A1E3-001D09F25217.root',
    #    '/store/relval/CMSSW_3_4_0_pre5/RelValBeamHalo/GEN-SIM-RECO/STARTUP3X_V11-v1/0001/083E5183-70CB-DE11-8D28-0030487A1990.root'
    

    # 326 Reco BeamHalo Relval
    
#    '/store/relval/CMSSW_3_2_6/RelValBeamHalo/GEN-SIM-RECO/STARTUP31X_V7-v1/0013/964133E3-FA9A-DE11-A228-0030487A18A4.root',
#    '/store/relval/CMSSW_3_2_6/RelValBeamHalo/GEN-SIM-RECO/STARTUP31X_V7-v1/0012/C2BD8B15-239A-DE11-98B6-001D09F29597.root',
#    '/store/relval/CMSSW_3_2_6/RelValBeamHalo/GEN-SIM-RECO/STARTUP31X_V7-v1/0012/B4D814F9-219A-DE11-BF6D-001D09F25109.root',
#    '/store/relval/CMSSW_3_2_6/RelValBeamHalo/GEN-SIM-RECO/STARTUP31X_V7-v1/0012/A23604E8-249A-DE11-8CD7-001617E30CD4.root',
    )
    )

process.p = cms.Path(process.BeamHaloId*process.AnalyzeBeamHalo)
#process.p = cms.Path(process.CSCHaloData*process.EcalHaloData*process.HcalHaloData*process.GlobalHaloData)
#process.p = cms.Path(process.CSCHaloData*process.HcalHaloData)


                                         
                                     
