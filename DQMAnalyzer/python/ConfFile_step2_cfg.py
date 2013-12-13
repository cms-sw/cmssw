import FWCore.ParameterSet.Config as cms

process = cms.Process("STEP2")

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("DQMServices.Components.EDMtoMEConverter_cfi")
process.load("DQMServices.Core.DQM_cfg")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')

process.load("GEMCode.DQMAnalyzer.CfiFile_step2_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(

        'file:first_200GeV.root',

    )
)

process.p = cms.Path(process.EDMtoMEConverter * process.DQMGEMSecondStep)
process.DQM.collectorHost = ''
process.DQM.collectorPort = 9090
process.DQM.debug = False
