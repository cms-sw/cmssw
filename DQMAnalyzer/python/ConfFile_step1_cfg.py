import FWCore.ParameterSet.Config as cms

process = cms.Process("STEP1")

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("DQMServices.Core.DQM_cfg")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')

process.load("GEMCode.DQMAnalyzer.CfiFile_step1_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(

	'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt200_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC1_v2/calabria_SingleMuPt200_RECOGEM_STDRECO_CMSSW_6_2_0_SLHC1/9d4158c7f7160e8742c9656b5dc7afdf/out_reco_std_100_1_cSy.root',

    )
)

process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *_MEtoEDMConverter_*_*'),
    fileName = cms.untracked.string('first_200GeV.root')
)

process.p = cms.Path(process.GLBMuonAnalyzerWithGEMs * process.MEtoEDMConverter)
process.outpath = cms.EndPath(process.FEVT)

process.DQM.collectorHost = ''
process.DQM.collectorPort = 9090
process.DQM.debug = False
