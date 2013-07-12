import FWCore.ParameterSet.Config as cms


process = cms.Process("test")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('HSCP Secondary Dataset'),
    name = cms.untracked.string('$Source: /cvs/CMSSW/CMSSW/Configuration/Skimming/test/test_HSCP_SD.py,v $')
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.GeometryExtended_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.EventContent.EventContent_cff')
process.GlobalTag.globaltag = "GR_P_V17::All"  


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v2/000/163/668/6CC98438-1B74-E011-A576-001617E30D0A.root',
    '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v2/000/163/668/5A7B6738-1B74-E011-B7E5-001617E30F58.root',
    '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v2/000/163/668/583F5FE7-1B74-E011-9DC2-001617C3B5E4.root',
    '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v2/000/163/668/2CA1FB84-1A74-E011-AC47-001617DC1F70.root',
    '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v2/000/163/668/247E2EAF-1A74-E011-9C7E-000423D94494.root'
    )
)

process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")

process.load("Configuration.Skimming.PDWG_HSCP_SD_cff")
process.hscpFilter = cms.Path(process.HSCPSD)

process.outputHSCP = cms.OutputModule("PoolOutputModule",
                                        dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('USER'),
        filterName = cms.untracked.string('SD_HSCP')),
                                        SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('hscpFilter')),                                        
                                        outputCommands = process.FEVTEventContent.outputCommands,
                                        fileName = cms.untracked.string('SD_HSCP_2011.root')
                                        )


process.this_is_the_end = cms.EndPath(
process.outputHSCP
)
