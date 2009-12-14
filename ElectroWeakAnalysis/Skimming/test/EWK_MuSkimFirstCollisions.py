import FWCore.ParameterSet.Config as cms

process = cms.Process("EWKMuSkimFirstCollisions")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# source

process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
     "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/04092AB7-75E8-DE11-958F-000423D98750.root",
     "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/04F15557-7BE8-DE11-8A41-003048D2C1C4.root",
     "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/3C02A810-7CE8-DE11-BB51-003048D375AA.root",
     "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/44255E49-80E8-DE11-B6DB-000423D991F0.root",
     "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/7C9741F5-78E8-DE11-8E69-001D09F2AD84.root",
     "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/EE9412FD-80E8-DE11-9FDD-000423D94908.root",
     "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/F08F782B-77E8-DE11-B1FC-0019B9F72BFF.root"

    )
)
# to handle some format problem  with some of the first CMS collsion runs
process.source.inputCommands = cms.untracked.vstring(
    "keep *",
    "drop L1GlobalTriggerObjectMapRecord_hltL1GtObjectMap__HLT"
    )


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.MessageLogger.cerr.threshold = ''
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('GR09_R_V1::All')
#process.GlobalTag.globaltag = cms.string('MC_31X_V3::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

# HLT filter
#import HLTrigger.HLTfilters.hltHighLevel_cfi
#process.EWK_MuHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
# Uncomment this to access 8E29 menu and filter on it
#process.EWK_MuHLTFilter.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT8E29")
#process.EWK_MuHLTFilter.HLTPaths = ["HLT_Mu3", "HLT_DoubleMu3"]
# Uncomment this to filter on 1E31 HLT menu
#process.EWK_MuHLTFilter.HLTPaths = ["HLT_Mu9", "HLT_DoubleMu3"]

process.load ('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('40 OR  41')

process.options = cms.untracked.PSet(
        SkipEvent = cms.untracked.vstring('ProductNotFound'),
            wantSummary = cms.untracked.bool(True)
        )

#  Merge CaloMuons into the collection of reco::Muons
from RecoMuon.MuonIdentification.calomuons_cfi import calomuons;
process.muons = cms.EDProducer("CaloMuonMerger",
    muons = cms.InputTag("muons"), # half-dirty thing. it works aslong as we're the first module using muons in the path
    caloMuons = cms.InputTag("calomuons"),
    minCaloCompatibility = calomuons.minCaloCompatibility)

## And re-make isolation, as we can't use the one in AOD because our collection is different
process.load('RecoMuon.MuonIsolationProducers.muIsolation_cff')



# Muon filter
process.goodMuons = cms.EDFilter("MuonViewRefSelector",
  src = cms.InputTag("muons"),
  cut = cms.string('pt > 0.5 && ( isGlobalMuon=1 || (isTrackerMuon =1  && numberOfMatches>=1 ) || isStandAloneMuon=1)'), # also || (isCaloMuon=1) ??
  filter = cms.bool(True)                                
)

# require at least two tracks with pt>.5, to hopefully remove further cosmic contaminations  
process.tracks = cms.EDFilter("TrackSelector",
  src=cms.InputTag("generalTracks"),
  cut = cms.string('pt > 0.5'),
  filter = cms.bool(True)                                
)

process.tracksFilter = cms.EDFilter("TrackCountFilter",
                                 src = cms.InputTag("tracks"),
                                 minNumber = cms.uint32(3)
                             )


process.dimuonsHLT = cms.EDFilter("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(False),
    cut = cms.string('mass > 0'),
    decay = cms.string('goodMuons@+ goodMuons@-')
)


# Output module configuration
from Configuration.EventContent.EventContent_cff import *
EWK_MuSkimEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)

EWK_MuSkimEventContent.outputCommands.extend(AODEventContent.outputCommands)


EWK_MuSkimEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring(
           'EWK_MuSkimPath')
    )
)


dimuonsEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'keep *_dimuonsAOD_*_*'
    )
 )


EWK_MuSkimEventContent.outputCommands.extend(dimuonsEventContent.outputCommands)



process.EWK_MuSkimOutputModule = cms.OutputModule("PoolOutputModule",
    EWK_MuSkimEventContent,
    EWK_MuSkimEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('EWKMuSkim'),
        dataTier = cms.untracked.string('USER')
   ),
   fileName = cms.untracked.string('testEWKMuSkim_L1TG4041AllMuAtLeastThreeTracks124120.root')
)



# Skim path
process.EWK_MuSkimPath = cms.Path(
 # process.EWK_MuHLTFilter +
  process.hltLevel1GTSeed+
  process.muons *
  process.muIsolation *
  process.goodMuons +
  process.tracks +
  process.tracksFilter +
  process.dimuonsHLT
)


process.outpath = cms.EndPath(process.EWK_MuSkimOutputModule)



