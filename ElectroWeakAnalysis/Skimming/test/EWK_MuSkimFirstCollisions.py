import FWCore.ParameterSet.Config as cms

process = cms.Process("EWKMuSkimFirstCollisions")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# source

process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
    
###  files at 

##

"rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/446/FE7241BD-EC49-DF11-8E75-001D09F29146.root",


    )
)
# to handle some format problem  with some of the first CMS collsion runs

process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*", "drop L1GlobalTriggerObjectMapRecord_hltL1GtObjectMap__HLT")



process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.MessageLogger.cerr.threshold = ''
process.MessageLogger.cerr.FwkReport.reportEvery = 10000

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('GR_R_35X_V6::All')
#process.GlobalTag.globaltag = cms.string('MC_31X_V3::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

# HLT filter
import HLTrigger.HLTfilters.hltHighLevel_cfi
process.EWK_MuHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
# Uncomment this to access 8E29 menu and filter on it
process.EWK_MuHLTFilter.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
process.EWK_MuHLTFilter.HLTPaths = [
          "HLT_MinBiasBSC",
          "HLT_L1Mu", "HLT_L1MuOpen", "HLT_L1Mu20",
          "HLT_L2Mu9", "HLT_L2Mu11",
          "HLT_Mu5", "HLT_Mu9"
          ] 

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
process.goodMuons = cms.EDFilter("MuonSelector",
  src = cms.InputTag("muons"),
  cut = cms.string('pt > 10.0 && ( isGlobalMuon=1 || (isTrackerMuon =1  && numberOfMatches>=1 ))'), # also || (isCaloMuon=1) ??
  filter = cms.bool(True)                                
)

process.rmCosmicFromGoodMuons = cms.EDFilter("MuonSelector",
  src = cms.InputTag("goodMuons"),
  cut = cms.string('abs(innerTrack().dxy)<1.0'),
  filter = cms.bool(True)                                
)


process.goodMuonsPt15 = cms.EDFilter("MuonSelector",
  src = cms.InputTag("goodMuons"),
  cut = cms.string('(isGlobalMuon=1 || isTrackerMuon =1) &&  pt > 15.0'), 
  filter = cms.bool(False)                                
)

# Dump of interesting events, with mu pt>15
process.eventDump = cms.EDAnalyzer(
    "EventDumper",
 #    srcMuons = cms.InputTag("goodMuonsPt15")
    )


# require at least two tracks with pt>.5, to hopefully remove further cosmic contaminations  
process.tracks = cms.EDFilter("TrackSelector",
  src=cms.InputTag("generalTracks"),
  cut = cms.string('abs(dxy)<0.5 && pt > 0.5 && hitPattern().numberOfValidPixelHits>0'),
  filter = cms.bool(True)                                
)

process.tracksFilter = cms.EDFilter("TrackCountFilter",
                                 src = cms.InputTag("tracks"),
                                 minNumber = cms.uint32(1)
                             )


process.dimuonsAOD = cms.EDFilter("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(False),
    cut = cms.string('mass > 0'),
    decay = cms.string('goodMuons@+ goodMuons@-')
)


# For creation of WMuNu Candidates
process.load("ElectroWeakAnalysis.WMuNu.wmunusProducer_cfi")




# Output module configuration
from Configuration.EventContent.EventContent_cff import *
EWK_MuSkimEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)

EWK_MuSkimEventContent.outputCommands.extend(FEVTEventContent.outputCommands)


EWK_MuSkimEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring(
           'EWK_MuSkimPath')
    )
)


dimuonsEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'keep *_dimuonsAOD_*_*',
    'keep *_CosmicFromGoodMuons_*_*', 
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
   fileName = cms.untracked.string('EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133446_1.root')

)



# Skim path
process.EWK_MuSkimPath = cms.Path(
  process.EWK_MuHLTFilter +
  process.hltLevel1GTSeed+
  process.goodMuons +
  process.rmCosmicFromGoodMuons +
#  process.tracks +
#  process.tracksFilter +
  process.dimuonsAOD +
  process.allWMuNus +
#  process.goodMuonsPt15 +
  process.eventDump


)


process.outpath = cms.EndPath(process.EWK_MuSkimOutputModule)

