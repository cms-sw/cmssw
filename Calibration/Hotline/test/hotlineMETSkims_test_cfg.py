import FWCore.ParameterSet.Config as cms

process = cms.Process("skim")

process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,'auto:run2_mc_GRun')

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
        fileNames = cms.untracked.vstring('root://eoscms:///eos/cms/store/relval/CMSSW_7_4_2/JetHT25ns/MINIAOD/GR_R_74_V12_19May_RelVal_jht25ns2012D-v1/00000/0AD399D0-C3FE-E411-817D-002618943896.root',
            'root://eoscms:///eos/cms/store/relval/CMSSW_7_4_2/JetHT25ns/MINIAOD/GR_R_74_V12_19May_RelVal_jht25ns2012D-v1/00000/3E4A21D0-C3FE-E411-B046-0026189438FA.root')
        #fileNames = cms.untracked.vstring('root://eoscms:///eos/cms/store/relval/CMSSW_7_5_0_pre1/RelValQCD_Pt_3000_3500_13/GEN-SIM-RECO/MCRUN2_74_V7-v1/00000/826FE575-5CE3-E411-BA9F-00248C0BE012.root')
)

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.load('Configuration.Geometry.GeometryExtended2015Reco_cff')

#hotline filters
process.load("Calibration.Hotline.hotlineMETSkims_cff")
process.load("Calibration.Hotline.hotlineMETSkims_Output_cff")

process.out = cms.OutputModule("PoolOutputModule",
    SelectEvents = process.OutALCARECOMETHotline.SelectEvents,
    outputCommands = process.OutALCARECOMETHotline.outputCommands,
    fileName = cms.untracked.string('provaMET.root')
)

process.e = cms.EndPath(process.out)
