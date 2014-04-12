import FWCore.ParameterSet.Config as cms


process = cms.Process("test")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('HWW central skim'),
    name = cms.untracked.string('$Source:  $')
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
    '/store/data/Run2011A/DoubleElectron/AOD/PromptReco-v1/000/161/312/ECDD3154-DF57-E011-9088-001D09F24D4E.root'
    # /store/data/Run2011A/SingleMu/AOD/PromptReco-v1/000/161/312/F8AEC745-DF57-E011-8D23-001D09F290BF.root'
    # /store/data/Run2011A/MuEG/AOD/PromptReco-v1/000/161/312/98EC336A-7959-E011-B557-0030487CD6B4.root
    ),
)

process.load("Configuration.Skimming.PDWG_HWWSkim_cff")
process.diMuonFilter = cms.Path(process.diMuonSequence)
process.diElectronFilter = cms.Path(process.diElectronSequence)
process.MuEleFilter = cms.Path(process.EleMuSequence)

process.outputCsHWW = cms.OutputModule("PoolOutputModule",
                                        dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('AOD'),
        filterName = cms.untracked.string('CS_HWW')),
                                        SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('diMuonFilter','diElectronFilter','MuEleFilter')),                                        
                                        outputCommands = process.FEVTEventContent.outputCommands,
                                        fileName = cms.untracked.string('CS_HWW_2011.root')
                                        )


process.this_is_the_end = cms.EndPath(
process.outputCsHWW
)
