import FWCore.ParameterSet.Config as cms

process = cms.Process("AlCaElectronsProduction")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('GR_R_52_V7::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalIsolElectron_cff")
#process.load("Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalIsolElectron_Output_cff")

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


readFiles = cms.untracked.vstring(    '/store/relval/CMSSW_5_2_4/SingleElectron/RECO/GR_R_52_V7_RelVal_wEl2011B_testPAxWMA-v1/0000/CCF2CBC1-469D-E111-B42B-003048679180.root'  )
secFiles = cms.untracked.vstring()
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)


#readFiles.extend(( ))



secFiles.extend( (
                   ) )


process.report = cms.EDAnalyzer("HLTrigReport",
    HLTriggerResults = cms.InputTag("TriggerResults","","HLT")
)

process.out = cms.OutputModule("PoolOutputModule",
    process.OutALCARECOEcalCalElectron,
    fileName = cms.untracked.string('testOutput.root')
)

process.pathALCARECOEcalCalElectron = cms.Path(process.report*process.seqALCARECOEcalCalDoubleElectron)

process.outpath = cms.EndPath(process.out)
#process.ewkHLTFilter.HLTPaths = ['HLT_DoubleEle10_Z' ]

#process.ewkHLTFilter.HLTPaths = ['HLT_Ele10_SW_L1R','HLT_IsoEle15_L1I', 'HLT_IsoEle15_LW_L1I','HLT_IsoEle18_L1R','HLT_IsoEle15_L1I','HLT_IsoEle18_L1R','HLT_IsoEle15_LW_L1I','HLT_LooseIsoEle15_LW_L1R','HLT_Ele10_SW_L1R','HLT_Ele15_SW_L1R','HLT_Ele15_LW_L1R','HLT_EM80','HLT_EM200','HLT_DoubleIsoEle10_L1I','HLT_DoubleIsoEle12_L1R','HLT_DoubleIsoEle10_LW_L1I','HLT_DoubleIsoEle12_LW_L1R','HLT_DoubleEle5_SW_L1R','HLT_DoubleEle10_LW_OnlyPixelM_L1R','HLT_DoubleEle10_Z','HLT_DoubleEle6_Exclusive']


