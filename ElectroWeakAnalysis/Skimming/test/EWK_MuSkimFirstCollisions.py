import FWCore.ParameterSet.Config as cms

process = cms.Process("EWKMuSkimFirstCollisions")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# source

process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
"rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/275/009FD47A-14EA-DE11-A16C-001D09F276CF.root",
"rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/275/04B52832-08EA-DE11-B565-001D09F23F2A.root",
"rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/275/10183EFB-14EA-DE11-AA9B-000423D6B444.root",
"rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/275/2015E326-19EA-DE11-9AB2-000423D99614.root",
"rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/275/40BA430D-17EA-DE11-B5B0-003048D2BE08.root",
"rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/275/44D74360-16EA-DE11-AB79-0030486730C6.root",
"rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/275/66C694A9-15EA-DE11-A243-001D09F2AD84.root",
"rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/275/6CA422E0-12EA-DE11-9F02-001D09F24399.root",
"rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/275/78BF3FE5-19EA-DE11-8C08-001D09F2AF1E.root",
"rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/275/7A01076A-0CEA-DE11-AA59-001D09F24600.root",
"rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/275/AE9C108A-13EA-DE11-86FF-001D09F2905B.root",
"rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/275/C22CBF0B-17EA-DE11-83E6-001617C3B77C.root",
"rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/275/C83608FB-14EA-DE11-9FE5-000423D98B08.root",
"rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/275/E8FBAD94-09EA-DE11-8FA9-001D09F2438A.root",
"rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/275/F0CEF7FA-14EA-DE11-BCB4-000423D99AA2.root",
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
process.GlobalTag.globaltag = cms.string('GR09_R_35X_V4::All')
#process.GlobalTag.globaltag = cms.string('MC_31X_V3::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

# HLT filter
import HLTrigger.HLTfilters.hltHighLevel_cfi
process.EWK_MuHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
# Uncomment this to access 8E29 menu and filter on it
process.EWK_MuHLTFilter.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
process.EWK_MuHLTFilter.HLTPaths=["HLT_MinBiasBSC_OR", "HLT_L1Mu", "HLT_L1MuOpen"] 
# Uncomment this to filter on 1E31 HLT menu
#process.EWK_MuHLTFilter.HLTPaths = ["HLT_Mu9", "HLT_DoubleMu3"]

process.load ('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('0 AND  (40 OR  41) AND NOT (36 OR 37 OR 38 OR 39)')

process.options = cms.untracked.PSet(
        SkipEvent = cms.untracked.vstring('ProductNotFound'),
            wantSummary = cms.untracked.bool(True)
        )

#  Merge CaloMuons into the collection of reco::Muons
#from RecoMuon.MuonIdentification.calomuons_cfi import calomuons;
#process.muons = cms.EDProducer("CaloMuonMerger",
#    muons = cms.InputTag("muons"), # half-dirty thing. it works aslong as we're the first module using muons in the path
#    caloMuons = cms.InputTag("calomuons"),
#    minCaloCompatibility = calomuons.minCaloCompatibility)

## And re-make isolation, as we can't use the one in AOD because our collection is different
#process.load('RecoMuon.MuonIsolationProducers.muIsolation_cff')



# Muon filter
process.goodMuons = cms.EDFilter("MuonViewRefSelector",
  src = cms.InputTag("muons"),
  cut = cms.string('pt > 1.0 && ( isGlobalMuon=1 || (isTrackerMuon =1  && numberOfMatches>=1 ) || isStandAloneMuon=1)'), # also || (isCaloMuon=1) ??
  filter = cms.bool(True)                                
)

# require at least two tracks with pt>.5, to hopefully remove further cosmic contaminations  
process.tracks = cms.EDFilter("TrackSelector",
  src=cms.InputTag("generalTracks"),
  cut = cms.string('abs(dxy)<0.5 && pt > 1.0 && hitPattern().numberOfValidPixelHits>0'),
  filter = cms.bool(True)                                
)

process.tracksFilter = cms.EDFilter("TrackCountFilter",
                                 src = cms.InputTag("tracks"),
                                 minNumber = cms.uint32(3)
                             )


process.dimuonsAOD = cms.EDFilter("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(False),
    cut = cms.string('mass > 0'),
    decay = cms.string('goodMuons@+ goodMuons@-')
)


# For creaton of WMuNu Candidates
process.load("ElectroWeakAnalysis.WMuNu.wmunusProducer_cfi")


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
    'keep *_dimuonsAOD_*_*',
    'keep recoWMuNuCandidates_*_*_*'
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
   fileName = cms.untracked.string('testEWKMuSkim_L1TG0_4041AllMuAtLeastThreeTracks124275.root')
)



# Skim path
process.EWK_MuSkimPath = cms.Path(
  process.EWK_MuHLTFilter +
  process.hltLevel1GTSeed+
  process.goodMuons +
  process.tracks +
  process.tracksFilter +
  process.dimuonsAOD +
  process.allWMuNus
)


process.outpath = cms.EndPath(process.EWK_MuSkimOutputModule)



