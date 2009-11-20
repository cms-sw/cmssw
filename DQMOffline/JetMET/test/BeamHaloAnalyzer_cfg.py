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
#process.GlobalTag.globaltag = 'STARTUP31X_V1::All'
process.GlobalTag.globaltag ='STARTUP31X_V7::All'
process.DQMStore = cms.Service("DQMStore")

process.load("Configuration/StandardSequences/ReconstructionCosmics_cff")

process.load("RecoMuon/Configuration/RecoMuon_cff")

process.load("DQMOffline/JetMET/BeamHaloAnalyzer_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
#process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.source = cms.Source("PoolSource",
                            debugFlag = cms.untracked.bool(True),
                            debugVebosity = cms.untracked.uint32(10),
                            fileNames = cms.untracked.vstring(
    
    #        '/store/relval/CMSSW_3_2_7/RelValZMM/GEN-SIM-RECO/STARTUP31X_V7-v1/0002/F811E48B-D9A8-DE11-8DE6-001D09F248FD.root'
    #    '/store/relval/CMSSW_3_2_6/RelValWjet_Pt_3000_3500/GEN-SIM-RECO/MC_31X_V8-v1/0013/5E72BEBB-B29A-DE11-9E51-001D09F252F3.root',
    #    '/store/relval/CMSSW_3_2_6/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V8-v1/0013/861B7E55-929A-DE11-8047-000423D944F8.root',
    #   '/store/relval/CMSSW_3_2_6/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V7-v1/0013/DAE4D3B7-4E9A-DE11-92B1-000423D6B444.root',
    #    '/store/relval/CMSSW_3_2_6/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V8-v1/0013/5AED20CA-4D9A-DE11-A2E6-000423D6B48C.root'
    #    '/store/relval/CMSSW_3_2_6/RelValWE/GEN-SIM-RECO/STARTUP31X_V7-v1/0013/82BD7D76-D39A-DE11-BF88-001D09F24934.root',
    
    # 326 Reco BeamHalo Relval
    
    '/store/relval/CMSSW_3_2_6/RelValBeamHalo/GEN-SIM-RECO/STARTUP31X_V7-v1/0013/964133E3-FA9A-DE11-A228-0030487A18A4.root',
    '/store/relval/CMSSW_3_2_6/RelValBeamHalo/GEN-SIM-RECO/STARTUP31X_V7-v1/0012/C2BD8B15-239A-DE11-98B6-001D09F29597.root',
    '/store/relval/CMSSW_3_2_6/RelValBeamHalo/GEN-SIM-RECO/STARTUP31X_V7-v1/0012/B4D814F9-219A-DE11-BF6D-001D09F25109.root',
    '/store/relval/CMSSW_3_2_6/RelValBeamHalo/GEN-SIM-RECO/STARTUP31X_V7-v1/0012/A23604E8-249A-DE11-8CD7-001617E30CD4.root',
    )
                            )
process.p = cms.Path(process.CSCHaloData*process.EcalHaloData*process.HcalHaloData*process.GlobalHaloData*process.BeamHaloSummary*process.AnalyzeBeamHalo)

#### If cosmic muons are not by default in the event, then you should run this sequence
#process.p = cms.Path(process.ctfWithMaterialTracksP5LHCNavigation*process.muonRecoLHC*process.CSCHaloData*process.EcalHaloData*process.HcalHaloData*process.GlobalHaloData*process.AnalyzeBeamHalo)
