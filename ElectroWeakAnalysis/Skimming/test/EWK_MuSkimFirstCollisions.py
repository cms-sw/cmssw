import FWCore.ParameterSet.Config as cms

process = cms.Process("EWKMuSkimFirstCollisions")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# source

process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(

###  files at 


###drwxrwxr-x  13 5410     zh                          0 Apr 14 09:38 158
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/22739669-9247-DF11-A523-000423D98634.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/2890DFEF-9547-DF11-A7A8-001D09F24EE3.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/46F468A2-9647-DF11-94DF-001D09F25109.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/58C5D01E-9347-DF11-B666-001D09F2514F.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/92DD947F-9447-DF11-8189-001D09F23A34.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/94B4CEA3-8F47-DF11-B3CF-0030487CD6D2.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/94D0EF59-9747-DF11-B6CE-001D09F24EAC.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/9C0D59D2-9347-DF11-B046-001D09F25041.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/AC9054B9-9147-DF11-90DA-001D09F244DE.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/D6EDC6BF-9147-DF11-944B-001D09F29169.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/DCDA20A7-8F47-DF11-AB42-001D09F2B30B.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/E882AAFE-9047-DF11-A604-001D09F2441B.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/D247EC06-9847-DF11-BD11-000423D9A212.root"

###drwxrwxr-x  26 5410     zh                          0 Apr 14 10:24 158


## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/3232CCEB-9C47-DF11-894E-0030487A3C92.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/36E7E6EA-9C47-DF11-85A3-0030487C90C2.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/4A3A1740-9C47-DF11-90EC-0030487C5CFA.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/4E63F2EB-9C47-DF11-80E4-00304879FBB2.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/6059EB5C-9E47-DF11-AE80-0030487A1990.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/666DA3EB-9C47-DF11-A5A1-0030487A1FEC.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/765D67BD-9847-DF11-917D-001D09F34488.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/84C435F7-9847-DF11-8D8F-001D09F24DDF.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/8A87833C-9C47-DF11-ADED-000423D99614.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/8E0783AD-9D47-DF11-A292-001D09F24D4E.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/C03FC15E-9E47-DF11-85CE-0030487A1FEC.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/C6D3195D-9E47-DF11-A889-0030487A3C92.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/268E9DED-9C47-DF11-BDAE-0030487A1990.root",

###drwxrwxr-x  42 5410     zh                          0 Apr 14 11:13 158


## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/02BCA276-A047-DF11-AA40-00304879EE3E.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/1A254D8A-A247-DF11-8FF7-00304879FC6C.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/1CED4247-A347-DF11-A3F2-001D09F27067.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/34B12EE0-A147-DF11-9050-001D09F290BF.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/5EC52D47-A347-DF11-8BCA-001D09F2516D.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/8A190248-A347-DF11-AE68-001D09F24498.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/9A8D40C5-9F47-DF11-B3F9-001D09F25456.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/B29DC576-A047-DF11-B232-0030487A17B8.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/F8E90F47-A347-DF11-8D1A-001D09F2AD4D.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/FACCE6C5-9F47-DF11-9FD7-001D09F282F5.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/02EED4AE-A447-DF11-AF58-001D09F29597.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/C00DB5F6-A347-DF11-90B9-001D09F28755.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/CA8750AE-A447-DF11-B740-001D09F297EF.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/E855C8F5-A347-DF11-AD92-0030487C8CBE.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/02EED4AE-A447-DF11-AF58-001D09F29597.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/CA8750AE-A447-DF11-B740-001D09F297EF.root"



###drwxrwxr-x  56 5410     zh                          0 Apr 14 12:26 158



## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/58FDA819-A647-DF11-85E5-001D09F23C73.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/62FCF758-A547-DF11-BA06-001D09F252E9.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/78568499-A947-DF11-85D6-000423D99CEE.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/8664DE59-A547-DF11-A7BF-001D09F242EA.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/8686CDC4-A647-DF11-8DA5-0030487A17B8.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/8A39A993-A947-DF11-B377-001D09F24F65.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/0850DDAF-AB47-DF11-A728-000423D98B6C.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/28D13F7E-AC47-DF11-8310-001D09F252E9.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/44612E65-AD47-DF11-A385-000423D98C20.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/9836DD7E-AC47-DF11-8DAD-001D09F24024.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/C439DC7D-AC47-DF11-AA80-001D09F2447F.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/D80D4D6A-AD47-DF11-A9B2-0030487CD162.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/E44F4A5B-AD47-DF11-90B5-000423D98B5C.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/76BF8983-AD47-DF11-87EE-0030487CD718.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/B0C13A53-AF47-DF11-B617-000423D94E1C.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/4038A8E0-A847-DF11-A8A2-0030487A17B8.root",


###drwxrwxr-x  81 5410     zh                          0 Apr 14 13:19 158


## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/266CBD4E-B147-DF11-AB20-000423D99996.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/26C7DE4F-B147-DF11-8076-0030487CD718.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/326B0B67-B347-DF11-B5AA-000423D991D4.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/40EB46BC-B247-DF11-81D2-0030487CD812.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/4A28674C-B147-DF11-A775-000423D985B0.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/820F7500-B247-DF11-BF91-000423D9A2AE.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/92DF67B9-B247-DF11-AE0F-0030487CD6D2.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/943ADB11-B247-DF11-8A92-0030487CD16E.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/A4966367-B347-DF11-8633-000423D99614.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/ACA348BE-B247-DF11-9425-000423D98BC4.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/ACE3FE4B-B147-DF11-8D26-000423D98930.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/B0C13A53-AF47-DF11-B617-000423D94E1C.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/B0CF264D-B147-DF11-B7CC-0030487C7828.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/B8DCB538-B347-DF11-9496-0030487CD16E.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/D6A45875-B147-DF11-8B51-0030487CD6B4.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/0004D6AB-B547-DF11-905A-0030487CD16E.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/06D326D2-B447-DF11-93AA-0030487CD6DA.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/1EE3E923-B447-DF11-B10B-000423D9890C.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/58142639-B647-DF11-BA0B-000423D98BC4.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/642FFED1-B447-DF11-AF0A-000423D98844.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/AE09B646-B447-DF11-B49A-0030487CD6B4.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/2CA339EF-B647-DF11-BA89-000423D98B5C.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/6E9F29EE-B647-DF11-B1D9-000423D98AF0.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/9EFC4AEA-B647-DF11-99C2-000423D98E6C.root",
## "rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/18B70AB9-B247-DF11-B9F3-0030487CD716.root",

##drwxrwxr-x  83 5410     zh                          0 Apr 14 13:29 158

###"rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/127DCEA3-B747-DF11-82AE-0030487CD700.root",
###"rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/78CF7165-B847-DF11-AA59-000423D99160.root"

###drwxrwxr-x  85 5410     zh                          0 Apr 14 14:59 158


"rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/26B05640-C447-DF11-868A-0030487CD906.root",
"rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v8/000/133/158/1AADA4F1-C447-DF11-831D-0030487CAF0E.root"

    
    )
)
# to handle some format problem  with some of the first CMS collsion runs
## process.source.inputCommands = cms.untracked.vstring(
##     "keep *",
##     "drop L1GlobalTriggerObjectMapRecord_hltL1GtObjectMap__HLT"
##     )

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
   fileName = cms.untracked.string('EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133158_6.root')
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
#  process.goodMuonsPt15 +
  process.eventDump


)


process.outpath = cms.EndPath(process.EWK_MuSkimOutputModule)

