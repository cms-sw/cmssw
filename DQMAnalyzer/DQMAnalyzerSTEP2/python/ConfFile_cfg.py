import FWCore.ParameterSet.Config as cms

process = cms.Process("STEP2")

process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("DQMServices.Components.EDMtoMEConverter_cfi")
process.load("DQMServices.Core.DQM_cfg")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')

process.load("DQMAnalyzer.DQMAnalyzerSTEP2.CfiFile_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(

        'file:/lustre/cms/store/user/calabria/DQMGem/SingleMuPt5/edm_SingleMuPt5_1.root',
        'file:/lustre/cms/store/user/calabria/DQMGem/SingleMuPt10/edm_SingleMuPt10_1.root',
        'file:/lustre/cms/store/user/calabria/DQMGem/SingleMuPt50/edm_SingleMuPt50_1.root',
        'file:/lustre/cms/store/user/calabria/DQMGem/SingleMuPt100/edm_SingleMuPt100_1.root',
        'file:/lustre/cms/store/user/calabria/DQMGem/SingleMuPt200/edm_SingleMuPt200_1.root',
        'file:/lustre/cms/store/user/calabria/DQMGem/SingleMuPt500/edm_SingleMuPt500_1.root',
        'file:/lustre/cms/store/user/calabria/DQMGem/SingleMuPt1000/edm_SingleMuPt1000_1.root',

    )
)

process.p = cms.Path(process.EDMtoMEConverter * process.DQMGEMSecondStep)
process.DQM.collectorHost = ''
process.DQM.collectorPort = 9090
process.DQM.debug = False
