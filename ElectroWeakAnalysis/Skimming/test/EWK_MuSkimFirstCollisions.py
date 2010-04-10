import FWCore.ParameterSet.Config as cms

process = cms.Process("EWKMuSkimFirstCollisions")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# source

process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(

###  files at 18:30

"rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/132/968/168948BD-B744-DF11-A4E8-0030487A3C9A.root",
"rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/132/968/22EF0D53-BD44-DF11-AE35-0030487A18F2.root",
"rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/132/968/248A55D1-B944-DF11-970A-0030487CD7C6.root",
"rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/132/968/2E2D7B48-B644-DF11-A8B8-0030487CD7C6.root",
"rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/132/968/2EFC219A-BC44-DF11-A1D3-0030487CD6E6.root",
"rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/132/968/30AECA94-B544-DF11-B704-000423D98EA8.root",
"rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/132/968/32F2902F-B444-DF11-A6A8-0030487CD180.root",
"rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/132/968/3657A301-B744-DF11-B091-00304879BAB2.root",
"rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/132/968/48B4C9BB-B744-DF11-8A62-00304879FA4A.root",
"rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/132/968/6807CCE4-B444-DF11-BC8F-0030487C8CB8.root",
"rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/132/968/94B4ECCC-B944-DF11-B658-0030487CD162.root",
"rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/132/968/9E3D6030-B444-DF11-9855-0030487CD7C0.root",
"rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/132/968/A2655C4C-B644-DF11-BB9F-0030487C635A.root",
"rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/132/968/A807DED3-B944-DF11-9F4D-0030487C6A66.root",
"rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/132/968/C63058CC-B944-DF11-B541-0030487CD7C0.root",
"rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/132/968/CC550189-BA44-DF11-89BA-0030487D0D3A.root",
"rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/132/968/D017E593-B544-DF11-9F04-001617C3B5F4.root",
"rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/132/968/D4234400-B744-DF11-8A1C-0030487C8CB8.root",
"rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/132/968/D844C567-B844-DF11-A0CC-0030487C5CFA.root",
"rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/132/968/EA497C88-BA44-DF11-B1A8-0030487CD700.root",
"rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/132/968/F64CC0CC-B944-DF11-9F19-0030487CAF0E.root",

    )
)
# to handle some format problem  with some of the first CMS collsion runs
process.source.inputCommands = cms.untracked.vstring(
    "keep *",
    "drop L1GlobalTriggerObjectMapRecord_hltL1GtObjectMap__HLT"
    )


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
process.EWK_MuHLTFilter.HLTPaths=["HLT_MinBiasBSC", "HLT_L1Mu", "HLT_L1MuOpen"] 
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
     srcMuons = cms.InputTag("goodMuonsPt15")
    )


# require at least two tracks with pt>.5, to hopefully remove further cosmic contaminations  
process.tracks = cms.EDFilter("TrackSelector",
  src=cms.InputTag("generalTracks"),
  cut = cms.string('abs(dxy)<0.5 && pt > 0.5 && hitPattern().numberOfValidPixelHits>0'),
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
   fileName = cms.untracked.string('EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132968_1.root')
#   fileName = cms.untracked.string('test.root')
)



# Skim path
process.EWK_MuSkimPath = cms.Path(
  process.EWK_MuHLTFilter +
  process.hltLevel1GTSeed+
  process.goodMuons +
  process.rmCosmicFromGoodMuons +
  process.tracks +
  process.tracksFilter +
  process.dimuonsAOD +
  process.allWMuNus +
  process.goodMuonsPt15 +
  process.eventDump


)


process.outpath = cms.EndPath(process.EWK_MuSkimOutputModule)

