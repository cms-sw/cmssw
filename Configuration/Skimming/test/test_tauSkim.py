import FWCore.ParameterSet.Config as cms


process = cms.Process("test")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('Tau central skim'),
    name = cms.untracked.string('$Source: /cvs/CMSSW/CMSSW/Configuration/Skimming/test/CSmaker_Tau_PDTau_35e29_cfg.py,v $')
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.GeometryExtended_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.EventContent.EventContent_cff')
process.GlobalTag.globaltag = "GR_P_V16::All"  


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v1/000/161/312/D00ECD06-FA57-E011-B670-003048F1BF66.root',
        '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v1/000/161/312/B4AA6B6D-E857-E011-A081-001617C3B76A.root',
        '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v1/000/161/312/9E43C5BB-0858-E011-9400-0030487C6A66.root',
        '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v1/000/161/312/9483279C-E557-E011-B828-001D09F291D2.root',
        '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v1/000/161/312/5A98F600-FA57-E011-88EE-003048F11C5C.root',
        '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v1/000/161/312/286081CA-E257-E011-8E2D-001D09F253D4.root'

        ),
                            secondaryFileNames = cms.untracked.vstring(

        '/store/data/Run2011A/SingleElectron/RAW/v1/000/161/312/B49B80DE-FA55-E011-89A5-001D09F2527B.root',
        '/store/data/Run2011A/SingleElectron/RAW/v1/000/161/312/9CAED4DC-EE55-E011-92E0-0030487CD6E6.root',
        '/store/data/Run2011A/SingleElectron/RAW/v1/000/161/312/98352641-0656-E011-A1F2-001617DBD472.root',
        '/store/data/Run2011A/SingleElectron/RAW/v1/000/161/312/86618229-DA55-E011-A93E-003048F118DE.root',
        '/store/data/Run2011A/SingleElectron/RAW/v1/000/161/312/48B3C7E4-CF55-E011-8551-000423D9A2AE.root',
        '/store/data/Run2011A/SingleElectron/RAW/v1/000/161/312/30FA984D-E455-E011-920E-003048F11C5C.root'
        )
)

process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")

process.load("Configuration.Skimming.PDWG_TauSkim_cff")
process.tauFilter = cms.Path(process.tauSkimSequence)

process.outputCsTau = cms.OutputModule("PoolOutputModule",
                                        dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('RAW-RECO'),
        filterName = cms.untracked.string('CS_Tau')),
                                        SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('tauFilter')),                                        
                                        outputCommands = process.FEVTEventContent.outputCommands,
                                        fileName = cms.untracked.string('CS_Tau_2011.root')
                                        )


process.this_is_the_end = cms.EndPath(
process.outputCsTau
)
