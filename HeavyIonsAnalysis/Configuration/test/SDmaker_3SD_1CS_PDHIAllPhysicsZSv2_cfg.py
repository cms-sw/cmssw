import FWCore.ParameterSet.Config as cms

process = cms.Process("makeSD")

# import of standard configurations
process.load("Configuration.StandardSequences.Services_cff")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Configuration.StandardSequences.ReconstructionHeavyIons_cff")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.EventContent.EventContentHeavyIons_cff')


process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.4 $'),
    annotation = cms.untracked.string('SD and central skims'),
    name = cms.untracked.string('$Source: /cvs/CMSSW/UserCode/icali/SkimsCfg/SDmaker_3SD_1CS_PDHIAllPhysicsZSv2_cfg.py,v $')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

#process.Timing = cms.Service("Timing")

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
     'file:/d101/icali/ROOTFiles_SWsrc392pa5/00DFBAEF-5741-E011-B023-0025901D6486.root',
     'file:/d101/icali/ROOTFiles_SWsrc392pa5/00151D16-8A41-E011-B938-003048CAAAB6.root',
     'file:/d101/icali/ROOTFiles_SWsrc392pa5/0005A206-A642-E011-8B41-000423D33970.root'
    #'/store/hidata/HIRun2010/HIAllPhysics/RECO/ZS-v2/0033/7E0F627F-5C43-E011-AF82-003048F1CA12.root'
    #'file:/d101/icali/ROOTFiles_SWsrc392pa5/3855C0DE-FCF4-DF11-857D-003048D2C092.root'
    #'/store/hidata/HIRun2010/HIAllPhysics/RAW/v1/000/151/878/FCB5F9D9-16F5-DF11-89B1-001D09F251FE.root'
))


# Other statements
process.GlobalTag.globaltag = "GR_R_39X_V6B::All"
process.MessageLogger.cerr.FwkReport.reportEvery = 100

############ Filters ###########
# BSC or HF coincidence (masked unprescaled L1 bits)
process.load('L1Trigger.Skimmer.l1Filter_cfi')
process.bscOrHfCoinc = process.l1Filter.clone(
   algorithms = cms.vstring('L1_BscMinBiasThreshold1', 'L1_HcalHfCoincidencePm')
)


# patch the collisionEventSelection
# make calotowers into candidates
process.towersAboveThreshold = cms.EDProducer("CaloTowerCandidateCreator",
                                              src = cms.InputTag("towerMaker"),
                                              verbose = cms.untracked.int32(0),
                                              minimumE = cms.double(3.0),
                                              minimumEt = cms.double(0.0),
                                              )

# select HF+ towers above threshold
process.hfPosTowers = cms.EDFilter("EtaPtMinCandSelector",
                                   src = cms.InputTag("towersAboveThreshold"),
                                   ptMin   = cms.double(0),
                                   etaMin = cms.double(3.0),
                                   etaMax = cms.double(6.0)
                                   )

# select HF- towers above threshold
process.hfNegTowers = cms.EDFilter("EtaPtMinCandSelector",
                                   src = cms.InputTag("towersAboveThreshold"),
                                   ptMin   = cms.double(0),
                                   etaMin = cms.double(-6.0),
                                   etaMax = cms.double(-3.0)
                                   )

# require at least one HF+ tower above threshold
process.hfPosFilter = cms.EDFilter("CandCountFilter",
                                   src = cms.InputTag("hfPosTowers"),
                                   minNumber = cms.uint32(1)
                                   )

# require at least one HF- tower above threshold
process.hfNegFilter = cms.EDFilter("CandCountFilter",
                                   src = cms.InputTag("hfNegTowers"),
                                   minNumber = cms.uint32(1)
                                   )

# three HF towers above threshold on each side
process.hfPosFilter3 = process.hfPosFilter.clone(minNumber=cms.uint32(3))
process.hfNegFilter3 = process.hfNegFilter.clone(minNumber=cms.uint32(3))

# Coincidence of HF towers above threshold
process.hfCoincFilter3 = cms.Sequence(
    process.towersAboveThreshold *
    process.hfPosTowers *
    process.hfNegTowers *
    process.hfPosFilter3 *
    process.hfNegFilter3)


# Selection of at least a two-track fitted vertex
process.primaryVertexFilter = cms.EDFilter("VertexSelector",
                                           src = cms.InputTag("hiSelectedVertex"),
                                           cut = cms.string("!isFake && abs(z) <= 25 && position.Rho <= 2 && tracksSize >= 2"), 
                                           filter = cms.bool(True),   # otherwise it won't filter the events
                                           )

# Cluster-shape filter re-run offline
process.load("RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi")

process.load("HLTrigger.special.hltPixelClusterShapeFilter_cfi")
process.hltPixelClusterShapeFilter.inputTag = "siPixelRecHits"

# Reject BSC beam halo L1 technical bits
from L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff import *
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import hltLevel1GTSeed
process.noBSChalo = hltLevel1GTSeed.clone(
    L1TechTriggerSeeding = cms.bool(True),
    L1SeedsLogicalExpression = cms.string('NOT (36 OR 37 OR 38 OR 39)')
    )

process.collisionEventSelection = cms.Sequence(process.noBSChalo *
                                               process.hfCoincFilter3 *
                                               process.primaryVertexFilter *
                                               process.siPixelRecHits *
                                               process.hltPixelClusterShapeFilter)



process.load("HLTrigger.HLTfilters.hltHighLevel_cfi")

### JetHI SD
process.hltJetHI = process.hltHighLevel.clone(HLTPaths = ['HLT_HIJet35U'])
process.filterSdJetHI = cms.Path(process.bscOrHfCoinc *
                                 process.collisionEventSelection *
                                 process.hltJetHI)

### PhotonHI SD
process.hltPhotonHI = process.hltHighLevel.clone(HLTPaths = ['HLT_HIPhoton15'])
process.filterSdPhotonHI = cms.Path(process.bscOrHfCoinc *
                                    process.collisionEventSelection *
                                    process.hltPhotonHI)

#process.hltPhotonJetHI = process.hltHighLevel.clone(
#  HLTPaths = ['HLT_HIJet35U','HLT_HIPhoton15'])
#process.filterSdPhotonJetHI = cms.Path(process.hltPhotonJetHI)

### MuHI SD

#process.hltDoubleMuHI = process.hltHighLevel.clone(HLTPaths = ['HLT_HIL1DoubleMuOpen'])
#process.filterSdDoubleMuHI = cms.Path(process.hltDoubleMuHI)

process.hltMuHI = process.hltHighLevel.clone(
  HLTPaths = ["HLT_HIL1DoubleMuOpen","HLT_HIL2DoubleMu0","HLT_HIL2DoubleMu3",
              "HLT_HIL1SingleMu3","HLT_HIL1SingleMu5","HLT_HIL1SingleMu7",
              "HLT_HIL2Mu20","HLT_HIL2Mu3","HLT_HIL2Mu5Tight"],
  throw = False,
  andOr = True)
process.filterSdMuHI = cms.Path(process.bscOrHfCoinc *
                                process.collisionEventSelection *
                                process.hltMuHI)

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
