import FWCore.ParameterSet.Config as cms


process = cms.Process("makeSD")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('Onia central skim'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/Skimming/test/CSmaker_Onia_PDMu_1e29_reprocess370patch2_cfg.py,v $')
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.GeometryExtended_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.EventContent.EventContent_cff')
process.GlobalTag.globaltag = "GR_R_37X_V6B::All"  


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/F85204EE-EB40-DF11-8F71-001A64789D1C.root'
        ),
                            secondaryFileNames = cms.untracked.vstring(
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/105/755/F6887FD0-9371-DE11-B69E-00304879FBB2.root'
        )
)
process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")

import HLTrigger.HLTfilters.hltHighLevelDev_cfi

### Onia skim CS
process.goodMuons = cms.EDFilter("MuonRefSelector",
    src = cms.InputTag("muons"),
    cut = cms.string("isGlobalMuon || (isTrackerMuon && numberOfMatches('SegmentAndTrackArbitration')>0)"),
)
process.diMuons = cms.EDProducer("CandViewShallowCloneCombiner",
    decay       = cms.string("goodMuons goodMuons"),
    checkCharge = cms.bool(False),
    cut         = cms.string("mass > 2"),
)
process.diMuonFilter = cms.EDFilter("CandViewCountFilter",
    src       = cms.InputTag("diMuons"),
    minNumber = cms.uint32(1),
)
process.Skim_diMuons = cms.Path(
    process.goodMuons    *
    process.diMuons      *
    process.diMuonFilter
)



process.outputCsOnia = cms.OutputModule("PoolOutputModule",
                                        dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('RAW-RECO'),
        filterName = cms.untracked.string('CS_Onia')),
                                        SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('Skim_diMuons')),                                        
                                        outputCommands = process.FEVTEventContent.outputCommands,
                                        fileName = cms.untracked.string('CS_Onia_1e29.root')
                                        )


process.this_is_the_end = cms.EndPath(process.outputCsOnia)
