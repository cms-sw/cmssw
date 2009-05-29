import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.TimerService = cms.Service("TimerService")

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)


#
#  DQM SERVICES
#
process.load("DQMServices.Core.DQM_cfg")

#
#  DQM SOURCES
#
process.load("CondCore.DBCommon.CondDBSetup_cfi")

#process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")

#process.load("Configuration.GlobalRuns.ReconstructionGR_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

#process.load("L1Trigger.Configuration.L1Config_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtBoardMapsConfig_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")
#process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1030.L1Menu2008_2E30_Unprescaled_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1030.L1Menu_2008MC_2E30_Unprescaled_cff")

#process.load("L1Trigger.HardwareValidation.L1HardwareValidation_cff")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.load("HLTriggerOffline.Common.FourVectorHLTriggerOffline_cfi")
process.load("HLTriggerOffline.Common.FourVectorHLTriggerOfflineClient_cfi")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")



process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)
process.source = cms.Source("PoolSource",

    #skipEvents = cms.untracked.uint32(3564),

    fileNames = 
#cms.untracked.vstring('file:test.root')
#cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre1/RelValMinBias/GEN-SIM-RECO/STARTUP_30X_v1/0001/48C7FFEB-49F4-DD11-85AB-001617C3B79A.root')
#cms.untracked.vstring('/store/relval/CMSSW_3_0_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0005/06F01F20-E9DD-DD11-956F-001617E30CA4.root')
#cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre1/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/065F4E0A-F7F7-DD11-BA61-001D09F2AF1E.root')

#cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre2/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0000/04A55351-8803-DE11-A538-001617E30D0A.root')
cms.untracked.vstring(
			 # 3_1_0_pre2
			 # ----------
       #'/store/relval/CMSSW_3_1_0_pre2/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0000/0C7DAE41-8A03-DE11-A686-001617DBCF90.root',
       #'/store/relval/CMSSW_3_1_0_pre2/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0000/3AC3EF48-8203-DE11-AC5A-001617DC1F70.root',
       #'/store/relval/CMSSW_3_1_0_pre2/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0000/3EFFB4F8-8103-DE11-9D4B-000423D98BE8.root',
       ##'/store/relval/CMSSW_3_1_0_pre2/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0000/6C0BD64D-8103-DE11-B010-00304879FA4C.root',
       ##'/store/relval/CMSSW_3_1_0_pre2/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0000/8AEB9029-8303-DE11-8C80-001D09F2960F.root',
       #'/store/relval/CMSSW_3_1_0_pre2/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0000/9C13439A-8003-DE11-9297-000423D174FE.root',
       #'/store/relval/CMSSW_3_1_0_pre2/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0001/9286C599-DB03-DE11-ADB3-000423D9970C.root'

       #'file:/data/rekovic/data_RelVal_3_1_0_pre2/MinBias/38C21702-DB03-DE11-B7E4-000423D99CEE.root',
       #'file:/data/rekovic/data_RelVal_3_1_0_pre2/MinBias/6094F299-4103-DE11-B49B-001617C3B6CE.root',
       #'file:/data/rekovic/data_RelVal_3_1_0_pre2/QCD_Pt_80_120/242DB962-5F03-DE11-A3C3-000423D6BA18.root',
       #'file:/data/rekovic/data_RelVal_3_1_0_pre2/QCD_Pt_80_120/8EA44632-5D03-DE11-B059-000423D99AAE.root',
       #'file:/data/rekovic/data_RelVal_3_1_0_pre2/QCD_Pt_80_120/E0B8C9A3-6103-DE11-BC18-001617DBCF6A.root',
       #'file:/data/rekovic/data_RelVal_3_1_0_pre2/QCD_Pt_80_120/F2D7CDB4-5C03-DE11-8C83-0019DB29C620.root',
       #'file:/data/rekovic/data_RelVal_3_1_0_pre2/QCD_Pt_80_120/FE9FBBD3-5D03-DE11-8F1A-000423D9853C.root',
       #'file:/data/rekovic/data_RelVal_3_1_0_pre2/TTbar/0C7DAE41-8A03-DE11-A686-001617DBCF90.root',
       #'file:/data/rekovic/data_RelVal_3_1_0_pre2/TTbar/3AC3EF48-8203-DE11-AC5A-001617DC1F70.root',
       #'file:/data/rekovic/data_RelVal_3_1_0_pre2/TTbar/3EFFB4F8-8103-DE11-9D4B-000423D98BE8.root',
       #'file:/data/rekovic/data_RelVal_3_1_0_pre2/TTbar/6C0BD64D-8103-DE11-B010-00304879FA4C.root',
       #'file:/data/rekovic/data_RelVal_3_1_0_pre2/TTbar/8AEB9029-8303-DE11-8C80-001D09F2960F.root',
       #'file:/data/rekovic/data_RelVal_3_1_0_pre2/TTbar/9286C599-DB03-DE11-ADB3-000423D9970C.root',
       #'file:/data/rekovic/data_RelVal_3_1_0_pre2/TTbar/9C13439A-8003-DE11-9297-000423D174FE.root'

			 # 3_1_0_pre4
			 # ----------
			 # TTbar
       #'/store/relval/CMSSW_3_1_0_pre4/RelValProdTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0002/38D39540-C915-DE11-9BBB-000423D94990.root',
       #'/store/relval/CMSSW_3_1_0_pre4/RelValProdTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0002/4E0FA53B-CC15-DE11-84AF-001617E30D40.root',
       #'/store/relval/CMSSW_3_1_0_pre4/RelValProdTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0002/50AEBD55-CB15-DE11-9394-001617C3B654.root',
       #'/store/relval/CMSSW_3_1_0_pre4/RelValProdTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0002/A665ADF0-C915-DE11-914B-000423D987E0.root',
       #'/store/relval/CMSSW_3_1_0_pre4/RelValProdTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0003/D0941BB9-5216-DE11-B254-001617C3B78C.root',
       #'/store/relval/CMSSW_3_1_0_pre4/RelValProdTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0003/F6EDF12D-AB16-DE11-B639-0019DB29C5FC.root' 
			 # QCD_Pt_30_50
        #'/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_30_50/GEN-SIM-RECO/STARTUP_30X_v1/0001/02244BBA-3816-DE11-8F2C-001A92971B7C.root',
        #'/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_30_50/GEN-SIM-RECO/STARTUP_30X_v1/0001/186DC2DA-3C16-DE11-80A3-00304867BFC6.root',
        #'/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_30_50/GEN-SIM-RECO/STARTUP_30X_v1/0001/34FC9329-3A16-DE11-8838-0018F3D096CE.root',
        #'/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_30_50/GEN-SIM-RECO/STARTUP_30X_v1/0001/BE028682-3B16-DE11-9280-001731AF65CF.root',
        #'/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_30_50/GEN-SIM-RECO/STARTUP_30X_v1/0001/C6226B97-B916-DE11-8918-001A92971B8C.root',
        #'/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_30_50/GEN-SIM-RECO/STARTUP_30X_v1/0001/DEB4FDB2-4316-DE11-8C95-0018F34D0D62.root',
        #'/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_30_50/GEN-SIM-RECO/STARTUP_30X_v1/0001/FC12B88C-4716-DE11-95F3-001A92971B1A.root',
        #'/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_30_50/GEN-SIM-RECO/STARTUP_30X_v1/0002/3625743A-FF16-DE11-99CF-0017312B577F.root',
        #'/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_30_50/GEN-SIM-RECO/STARTUP_30X_v1/0002/B8CF0AA0-5C18-DE11-95BE-0018F3D096A6.root'


        # MinBias 9k
        #'/store/relval/CMSSW_3_1_0_pre4/RelValProdMinBias/GEN-SIM-RECO/IDEAL_30X_v1/0002/182491FD-AD15-DE11-8A6E-000423D99658.root',
        #'/store/relval/CMSSW_3_1_0_pre4/RelValProdMinBias/GEN-SIM-RECO/IDEAL_30X_v1/0003/52B11E2D-AB16-DE11-A5FE-000423D9853C.root'

        # ZEE 9k
#       '/store/relval/CMSSW_3_1_0_pre4/RelValZEE/GEN-SIM-RECO/STARTUP_30X_v1/0003/5C15A821-F215-DE11-8B3E-000423D60FF6.root',
#       '/store/relval/CMSSW_3_1_0_pre4/RelValZEE/GEN-SIM-RECO/STARTUP_30X_v1/0003/5404B103-F315-DE11-8B61-000423D944F0.root',
#       '/store/relval/CMSSW_3_1_0_pre4/RelValZEE/GEN-SIM-RECO/STARTUP_30X_v1/0003/4465ADE1-EF15-DE11-A752-000423D987FC.root',
#       '/store/relval/CMSSW_3_1_0_pre4/RelValZEE/GEN-SIM-RECO/STARTUP_30X_v1/0003/1065FBC4-AB16-DE11-BB83-000423D6CAF2.root' , 



        # WE 9k
#        '/store/relval/CMSSW_3_1_0_pre4/RelValWE/GEN-SIM-RECO/STARTUP_30X_v1/0003/7A9E91A9-AB16-DE11-9273-000423D6B42C.root',
#        '/store/relval/CMSSW_3_1_0_pre4/RelValWE/GEN-SIM-RECO/STARTUP_30X_v1/0003/98B92948-0A16-DE11-B55C-001617C3B654.root',
#        '/store/relval/CMSSW_3_1_0_pre4/RelValWE/GEN-SIM-RECO/STARTUP_30X_v1/0003/B818F2AB-1316-DE11-99C2-000423D94534.root',
#        '/store/relval/CMSSW_3_1_0_pre4/RelValWE/GEN-SIM-RECO/STARTUP_30X_v1/0003/F6196F96-0816-DE11-B2CF-000423D6CA6E.root'

#       # ZMM 9k
#        '/store/relval/CMSSW_3_1_0_pre4/RelValZMM/GEN-SIM-RECO/STARTUP_30X_v1/0003/5C786CEA-D415-DE11-9F1D-000423D6B358.root',
#        '/store/relval/CMSSW_3_1_0_pre4/RelValZMM/GEN-SIM-RECO/STARTUP_30X_v1/0003/62F9AF48-E315-DE11-AF8E-001D09F24047.root',
#        '/store/relval/CMSSW_3_1_0_pre4/RelValZMM/GEN-SIM-RECO/STARTUP_30X_v1/0003/762338D4-6316-DE11-8D10-000423D991F0.root',
#        '/store/relval/CMSSW_3_1_0_pre4/RelValZMM/GEN-SIM-RECO/STARTUP_30X_v1/0003/AEDC409C-AB16-DE11-BE1D-001617E30E28.root'
         
        # WM
#       '/store/relval/CMSSW_3_1_0_pre4/RelValWM/GEN-SIM-RECO/STARTUP_30X_v1/0003/84A4CB2C-AC16-DE11-B24E-001617C3B654.root',
#       '/store/relval/CMSSW_3_1_0_pre4/RelValWM/GEN-SIM-RECO/STARTUP_30X_v1/0003/4A4F7573-5016-DE11-9CCD-000423D99E46.root',
#       '/store/relval/CMSSW_3_1_0_pre4/RelValWM/GEN-SIM-RECO/STARTUP_30X_v1/0002/86DA2C48-C115-DE11-B7AE-000423D98DB4.root',
#       '/store/relval/CMSSW_3_1_0_pre4/RelValWM/GEN-SIM-RECO/STARTUP_30X_v1/0002/5A99E575-BB15-DE11-90B3-000423D986C4.root'

#
#    
#        # Gamma Jets
#        '/store/relval/CMSSW_3_1_0_pre4/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0003/10F03E9C-AB16-DE11-8B8A-001617DBCF90.root',
#        '/store/relval/CMSSW_3_1_0_pre4/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0003/2263E110-E815-DE11-A299-001617E30D52.root',
#        '/store/relval/CMSSW_3_1_0_pre4/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0003/32CABD3B-E715-DE11-B32F-000423D985B0.root',
#        '/store/relval/CMSSW_3_1_0_pre4/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0003/A6785F3E-E915-DE11-9E2C-001D09F23A34.root',
#        '/store/relval/CMSSW_3_1_0_pre4/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0003/E011B7C2-5216-DE11-B449-000423D99B3E.root'


#####  RelVal 310_pre5
       # Single gamma 10

			 '/store/relval/CMSSW_3_1_0_pre5/RelValSingleGammaPt10/GEN-SIM-RECO/IDEAL_31X_v1/0000/6EFC1418-0C2C-DE11-94BB-000423D9890C.root'


)

#cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre1/RelValSingleMuPt100/GEN-SIM-RECO/IDEAL_30X_v1/0001/325025BD-49F4-DD11-92D4-001617C3B70E.root')
)

process.MessageLogger = cms.Service("MessageLogger",
    detailedInfo = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    critical = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    debugModules = cms.untracked.vstring('*'),
    #debugModules = cms.untracked.vstring('hltResults'),
    #debugModules = cms.untracked.vstring('none-blah'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        WARNING = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noLineBreaks = cms.untracked.bool(True)
    ),
    destinations = cms.untracked.vstring('detailedInfo', 
        'critical', 
        'cout')
)

process.psource = cms.Path(process.hltResults*process.HLTriggerOfflineFourVectorClient)
process.p = cms.EndPath(process.dqmSaver)
process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True


