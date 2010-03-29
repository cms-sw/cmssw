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
#   "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/0A47492A-3FE9-DE11-9038-000423D98950.root",
##    "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/0A52D8E9-3AE9-DE11-8776-000423D99AAE.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/0E430DC0-4EE9-DE11-99BD-000423D944F8.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/0E9C444C-46E9-DE11-9919-000423D33970.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/161064C8-3DE9-DE11-995B-000423D9989E.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/16D166DD-28E9-DE11-9B36-001617C3B79A.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/16F79901-47E9-DE11-8DAD-001D09F2924F.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/1AD95606-3BE9-DE11-AD9D-000423D998BA.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/22644AAF-47E9-DE11-ADC9-001617E30D40.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/22993BEE-4BE9-DE11-94E9-001D09F290BF.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/282F95E1-3FE9-DE11-8449-001D09F26C5C.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/2C58FB62-26E9-DE11-8EA1-001D09F23174.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/327A1164-23E9-DE11-A0D7-001617DBCF6A.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/34E532E4-28E9-DE11-AE49-000423D8FA38.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/38350D91-25E9-DE11-A1E4-001D09F2910A.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/3A8196ED-4BE9-DE11-8AEC-001D09F24682.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/3EC2537D-2DE9-DE11-899E-001D09F231C9.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/40E2AB23-44E9-DE11-9119-000423D98B6C.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/465B152E-2EE9-DE11-8245-001617C3B710.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/4837513C-29E9-DE11-B532-000423D98B5C.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/508D3170-48E9-DE11-8BF6-000423D990CC.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/52B3E16F-48E9-DE11-A029-000423D996C8.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/5801B43E-26E9-DE11-8E9B-001D09F251BD.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/584236D2-38E9-DE11-AD14-001617C3B76A.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/588B64F5-2AE9-DE11-9075-001D09F2932B.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/5EAE0940-35E9-DE11-BC3E-0030487D0D3A.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/7080B357-26E9-DE11-A699-001D09F2A690.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/7240D970-48E9-DE11-AA8C-003048D2BE08.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/76013040-35E9-DE11-BA0F-000423D98EC8.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/805AB14F-41E9-DE11-8C2F-000423D990CC.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/820E773F-3AE9-DE11-865E-001617DC1F70.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/86C55930-2EE9-DE11-A418-001617E30CE8.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/86D5ADB7-47E9-DE11-923B-001617E30D52.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/8A809D11-2CE9-DE11-999B-000423D987E0.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/8C0185FC-26E9-DE11-876B-000423D99EEE.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/8C6AC712-3DE9-DE11-934D-001D09F25109.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/9046B4D7-28E9-DE11-94C2-003048D374F2.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/921CDC25-49E9-DE11-AA16-000423D98BC4.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/942AB6EA-3AE9-DE11-9BAD-000423D6006E.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/9C818516-2CE9-DE11-ACB6-000423D9939C.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/A4C77206-47E9-DE11-9327-001D09F295A1.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/A65FDC6E-2BE9-DE11-84A5-0019DB29C614.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/A67930BE-2CE9-DE11-B2FC-0019B9F72CE5.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/BAEFFD93-45E9-DE11-AE70-000423D98BC4.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/C28D0DD2-38E9-DE11-96E1-001617E30D52.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/C8FC37FC-46E9-DE11-8F9F-001D09F29524.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/CA5E9131-2CE9-DE11-8C5A-000423D991D4.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/DE1BC110-3DE9-DE11-A75C-001D09F28F1B.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/F048C0F7-26E9-DE11-B916-000423D98F98.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/F4620413-2CE9-DE11-8B48-000423D98950.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/FC55FF2C-49E9-DE11-B432-000423D99996.root",
## "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/230/FE40A912-2CE9-DE11-BD1F-000423D99660.root"



##      "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/04092AB7-75E8-DE11-958F-000423D98750.root",
##      "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/04F15557-7BE8-DE11-8A41-003048D2C1C4.root",
##      "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/3C02A810-7CE8-DE11-BB51-003048D375AA.root",
##      "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/44255E49-80E8-DE11-B6DB-000423D991F0.root",
##      "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/7C9741F5-78E8-DE11-8E69-001D09F2AD84.root",
##      "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/EE9412FD-80E8-DE11-9FDD-000423D94908.root",
##      "rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/F08F782B-77E8-DE11-B1FC-0019B9F72BFF.root"

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
process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('0 AND  (40 OR  41) AND NOT (bit 36 OR bit 37 OR bit 38 OR bit 39)')

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
   fileName = cms.untracked.string('testEWKMuSkim_L1TG4041AllMuAtLeastThreeTracks124275.root')
)



# Skim path
process.EWK_MuSkimPath = cms.Path(
  process.EWK_MuHLTFilter +
  process.hltLevel1GTSeed+
  process.muons *
  process.muIsolation *
  process.goodMuons +
  process.tracks +
  process.tracksFilter +
  process.dimuonsAOD +
  process.allWmuNus
)


process.outpath = cms.EndPath(process.EWK_MuSkimOutputModule)



