import FWCore.ParameterSet.Config as cms

process = cms.Process("AlCaElectronsProduction")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('STARTUP_V4::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalIsolElectron_cff")
process.load("Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalIsolElectron_Output_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.options = cms.untracked.PSet(
        wantSummary = cms.untracked.bool(True)
        )

#process.source = cms.Source("PoolSource",
#    debugVerbosity = cms.untracked.uint32(1),
#    debugFlag = cms.untracked.bool(False),
#    fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_6/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0000/001AC8F1-C478-DD11-9348-001617E30F4C.root')
#)


readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)


readFiles.extend( (
                  '/store/relval/CMSSW_2_1_9/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/04419036-F385-DD11-B3A7-001617C3B6E8.root',
                  '/store/relval/CMSSW_2_1_9/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/0A28F869-F285-DD11-AF3C-001617DBD5B2.root',
                  '/store/relval/CMSSW_2_1_9/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/162C4B5E-F585-DD11-872A-001617C3B64C.root',
                  '/store/relval/CMSSW_2_1_9/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/205E6CE3-F485-DD11-9D53-001617C3B76A.root',
                  '/store/relval/CMSSW_2_1_9/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/562BAFA1-F585-DD11-B931-001617DBD224.root',
                  '/store/relval/CMSSW_2_1_9/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/565AFE10-EF85-DD11-8353-000423D6B42C.root',
                  '/store/relval/CMSSW_2_1_9/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/5C66302A-F185-DD11-81D3-000423D98834.root',
                  '/store/relval/CMSSW_2_1_9/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/66F60641-F685-DD11-A493-000423D987FC.root',
                  '/store/relval/CMSSW_2_1_9/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/6E6A6E2D-F485-DD11-B707-001617DBD472.root',
                  '/store/relval/CMSSW_2_1_9/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/70191C8D-F485-DD11-8280-001617E30D06.root',
                  '/store/relval/CMSSW_2_1_9/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/9A204B65-F385-DD11-9CF1-000423D98B6C.root',
                  '/store/relval/CMSSW_2_1_9/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/A6E8BAB0-F085-DD11-9AB1-000423D986C4.root',
                  '/store/relval/CMSSW_2_1_9/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/B4D8FA14-F485-DD11-A41D-001617C3B76A.root',
                  '/store/relval/CMSSW_2_1_9/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/B6CA9FAB-F185-DD11-B66B-001617E30D0A.root',
                  '/store/relval/CMSSW_2_1_9/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/D8DAE3BF-F385-DD11-9C8E-001617C3B65A.root',
                  '/store/relval/CMSSW_2_1_9/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/EA02E08F-F285-DD11-8AF3-000423D9870C.root',
                  '/store/relval/CMSSW_2_1_9/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0001/6E9B44E2-0487-DD11-BFA7-001617C3B78C.root') );


secFiles.extend( (
                   ) )


process.report = cms.EDAnalyzer("HLTrigReport",
    HLTriggerResults = cms.InputTag("TriggerResults","","HLT")
)

process.out = cms.OutputModule("PoolOutputModule",
    process.OutALCARECOEcalCalElectron,
    fileName = cms.untracked.string('Zee_219.root')
)

process.pathALCARECOEcalCalElectron = cms.Path(process.report*process.seqALCARECOEcalCalElectron)

process.outpath = cms.EndPath(process.out)
#process.ewkHLTFilter.HLTPaths = ['HLT_DoubleEle10_Z' ]

process.ewkHLTFilter.HLTPaths = ['HLT_Ele10_SW_L1R','HLT_IsoEle15_L1I', 'HLT_IsoEle15_LW_L1I','HLT_IsoEle18_L1R','HLT_IsoEle15_L1I','HLT_IsoEle18_L1R','HLT_IsoEle15_LW_L1I','HLT_LooseIsoEle15_LW_L1R','HLT_Ele10_SW_L1R','HLT_Ele15_SW_L1R','HLT_Ele15_LW_L1R','HLT_EM80','HLT_EM200','HLT_DoubleIsoEle10_L1I','HLT_DoubleIsoEle12_L1R','HLT_DoubleIsoEle10_LW_L1I','HLT_DoubleIsoEle12_LW_L1R','HLT_DoubleEle5_SW_L1R','HLT_DoubleEle10_LW_OnlyPixelM_L1R','HLT_DoubleEle10_Z','HLT_DoubleEle6_Exclusive']

process.goodSuperClusters.cut = 'et > 20'
process.goodSuperClusterFilter.minNumber = 1
process.goodSuperClusters2.cut = 'et > 15'
process.goodSuperClusterFilter2.minNumber = 2
process.goodElectrons.cut = 'et > 20'
process.goodElectronFilter.minNumber = 0
process.goodElectrons2.cut = 'et > 15'
process.goodElectronFilter2.minNumber = 0
process.testSelector.src = 'egammaElectronTkIsolation'
process.testSelector.max = 0.02
process.testSelector.filter = True
process.alCaIsolatedElectrons.eventWeight = 1.

