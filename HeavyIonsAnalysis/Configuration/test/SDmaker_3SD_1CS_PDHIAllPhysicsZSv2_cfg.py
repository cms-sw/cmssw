import FWCore.ParameterSet.Config as cms

process = cms.Process("makeSD")

# import of standard configurations
process.load("Configuration.StandardSequences.Services_cff")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.EventContent.EventContentHeavyIons_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5000)
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

#process.Timing = cms.Service("Timing")

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    'file:/d101/icali/ROOTFiles_SWsrc392pa5/3855C0DE-FCF4-DF11-857D-003048D2C092.root'
    #'/store/hidata/HIRun2010/HIAllPhysics/RAW/v1/000/151/878/FCB5F9D9-16F5-DF11-89B1-001D09F251FE.root'
))

# Other statements
process.GlobalTag.globaltag = "GR_R_39X_V6B::All"
process.MessageLogger.cerr.FwkReport.reportEvery = 100

############ Filters ###########

process.load("HLTrigger.HLTfilters.hltHighLevel_cfi")

### JetHI SD
process.hltJetHI = process.hltHighLevel.clone(HLTPaths = ['HLT_HIJet35U'])
process.filterSdJetHI = cms.Path(process.hltJetHI)

### PhotonHI SD
process.hltPhotonHI = process.hltHighLevel.clone(HLTPaths = ['HLT_HIPhoton15'])
process.filterSdPhotonHI = cms.Path(process.hltPhotonHI)

#process.hltPhotonJetHI = process.hltHighLevel.clone(
#  HLTPaths = ['HLT_HIJet35U','HLT_HIPhoton15'])
#process.filterSdPhotonJetHI = cms.Path(process.hltPhotonJetHI)

### MuHI SD

#process.hltDoubleMuHI = process.hltHighLevel.clone(HLTPaths = ['HLT_HIL1DoubleMuOpen'])
#process.filterSdDoubleMuHI = cms.Path(process.hltDoubleMuHI)

process.hltMuHI = process.hltHighLevel.clone(
  HLTPaths = ['HLT_HIL1DoubleMuOpen','HLT_HIL1SingleMu3','HLT_HIL2Mu3'],
  throw = False)
process.filterSdMuHI = cms.Path(process.hltMuHI)

############ Output Modules ##########

### JetHI SD
process.outputSdJetHI = cms.OutputModule("PoolOutputModule",
                                         SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring('filterSdJetHI')),                               
                                         dataset = cms.untracked.PSet(
    dataTier = cms.untracked.string('RECO'),
    filterName = cms.untracked.string('SD_JetHI')),
                                         outputCommands = process.RECOEventContent.outputCommands,
                                         fileName = cms.untracked.string('SD_JetHI.root')
                                         )

### PhotonHI SD
process.outputSdPhotonHI = cms.OutputModule("PoolOutputModule",
                                            SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring('filterSdPhotonHI')),                               
                                            dataset = cms.untracked.PSet(
    dataTier = cms.untracked.string('RECO'),
    filterName = cms.untracked.string('SD_PhotonHI')),
                                            outputCommands = process.RECOEventContent.outputCommands,
                                            fileName = cms.untracked.string('SD_PhotonHI.root')
                                            )

### MuHI SD
process.outputSdMuHI = cms.OutputModule("PoolOutputModule",
                                        SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring('filterSdMuHI')),                               
                                        dataset = cms.untracked.PSet(
    dataTier = cms.untracked.string('RECO'),
    filterName = cms.untracked.string('SD_MuHI')),
                                        outputCommands = process.RECOEventContent.outputCommands,
                                        fileName = cms.untracked.string('SD_MuHI.root')
                                        )


process.this_is_the_end = cms.EndPath(
    process.outputSdJetHI      +
    process.outputSdPhotonHI   +
    process.outputSdMuHI       
)
