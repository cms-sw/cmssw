import FWCore.ParameterSet.Config as cms

process = cms.Process("makeSD")

# import of standard configurations
process.load("Configuration.StandardSequences.Services_cff")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Configuration.StandardSequences.GeometryExtended_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load('Configuration.StandardSequences.ReconstructionHeavyIons_cff')
process.load('PhysicsTools.PatAlgos.patHeavyIonSequences_cff')
process.load("HeavyIonsAnalysis.Configuration.analysisFilters_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.EventContent.EventContentHeavyIons_cff')
process.load("HeavyIonsAnalysis.Configuration.analysisEventContent_cff")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.2 $'),
    annotation = cms.untracked.string('SD and central skims'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/HeavyIonsAnalysis/Configuration/test/SDmaker_3SD_3CS_PDMinBias_cfg.py,v $')
    )

process.Timing = cms.Service("Timing")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_8_0/RelValHydjetQ_B0_2760GeV/GEN-SIM-RECO/MC_38Y_V7-v1/0005/C85FD627-6895-DF11-B5CA-0026189437F5.root'
    ),
    inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")                  
)

# Other statements
#process.GlobalTag.globaltag = "MC_37Y_V4::All"
process.GlobalTag.globaltag = "MC_38Y_V8::All"
process.MessageLogger.cerr.FwkReport.reportEvery = 100

############ Filters ###########

### JetHI SD
process.filterSdJetHI = cms.Path(process.hltJetHI)

### PhotonHI SD
process.filterSdPhotonHI = cms.Path(process.hltPhotonHI)

### MuHI SD
process.filterSdMuHI = cms.Path(process.hltMuHI)

### JetHI AOD CS
process.filterCsJetHI = cms.Path(process.makeHeavyIonJets *
                                 process.selectedPatJets *
                                 process.countPatJets *
                                 process.kt4CaloJets *
                                 process.ak5CaloJets)

### MuHI AOD CS
process.filterCsMuHI = cms.Path(process.muonSelector *
                                process.muonFilter *
                                process.makeHeavyIonMuons)

### Zmumu AOD CS
process.filterCsZmumuHI = cms.Path(process.muonSelector *
                                   process.muonFilter *
                                   process.dimuonsMassCut *
                                   process.dimuonsMassCutFilter *
                                   process.makeHeavyIonMuons)


############ PAT specifics ###########

### for HI adaptations
from PhysicsTools.PatAlgos.tools.heavyIonTools import *
configureHeavyIons(process)

### disable MC
disableMonteCarloDeps(process)

############ Output modules ###########


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

### JetHI CS
process.outputCsJetHI = cms.OutputModule("PoolOutputModule",
                                        SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring('filterCsJetHI')),                               
                                        dataset = cms.untracked.PSet(
    dataTier = cms.untracked.string('RECO'),
    filterName = cms.untracked.string('CS_JetHI')),
                                        outputCommands = process.jetTrkSkimContent.outputCommands,
                                        fileName = cms.untracked.string('CS_JetHI.root')
                                        )

process.outputCsJetHI.outputCommands.extend([
    "keep recoVertexs_hiSelectedVertex__RECO",
    "keep recoTracks_hiGeneralTracks__RECO",
    "keep recoPhotons_*_*_*" ,
    "keep edmTriggerResults_TriggerResults__*" ,
    "keep triggerTriggerEvent_hltTriggerSummaryAOD__*"
    ])

### MuHI CS
process.outputCsMuHI = cms.OutputModule("PoolOutputModule",
                                        SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring('filterCsMuHI')),                               
                                        dataset = cms.untracked.PSet(
    dataTier = cms.untracked.string('RECO'),
    filterName = cms.untracked.string('CS_MuHI')),
                                        outputCommands = process.muonSkimContent.outputCommands,
                                        fileName = cms.untracked.string('CS_MuHI.root')
                                        )

### Z mumu  HI CS
process.outputCsZmumuHI  = cms.OutputModule("PoolOutputModule",
                                        SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring('filterCsZmumuHI')),                               
                                        dataset = cms.untracked.PSet(
    dataTier = cms.untracked.string('RECO'),
    filterName = cms.untracked.string('CS_ZmumuHI')),
                                        outputCommands = process.RECOEventContent.outputCommands,
                                        fileName = cms.untracked.string('CS_ZmumuHI.root')
                                        )

process.outputCsZmumuHI.outputCommands.extend(process.muonContent.outputCommands)
process.outputCsZmumuHI.outputCommands.extend(["keep *_dimuonsMassCut_*_*"])

process.this_is_the_end = cms.EndPath(
    process.outputSdJetHI      +
    process.outputSdPhotonHI   +
    process.outputSdMuHI       +
    process.outputCsJetHI      +
    process.outputCsMuHI       +
    process.outputCsZmumuHI
)
