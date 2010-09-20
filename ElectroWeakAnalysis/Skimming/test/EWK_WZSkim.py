import FWCore.ParameterSet.Config as cms

process = cms.Process("EWKWZSkim")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.options.FailPath = cms.untracked.vstring('ProductNotFound')

# source
process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
      'rfio:/castor/cern.ch/user/j/jalcaraz/Data2010/GoldenWmunus_132440-140182.root',
      'rfio:/castor/cern.ch/user/j/jalcaraz/Data2010/GoldenWmunus_140183-142557.root',
      'rfio:/castor/cern.ch/user/j/jalcaraz/Data2010/GoldenWmunus_142558-143179.root',
      'rfio:/castor/cern.ch/user/j/jalcaraz/Data2010/GoldenWmunus_143180-143336.root',
      'rfio:/castor/cern.ch/user/j/jalcaraz/Data2010/GoldenWmunus_143337-144114.root',
      'rfio:/castor/cern.ch/user/j/jalcaraz/Data2010/GoldenZmumus_132440-144114.root'
    ),
    #inputCommands = cms.untracked.vstring(
    #  'keep *',
    #  'drop *_MEtoEDMConverter_*_*',
    #  'drop *_lumiProducer_*_*',
    #  'drop *_*_*_HLT8E29',
    #  'drop edmTriggerResults_TriggerResults__*',
    #  'keep edmTriggerResults_TriggerResults__HLT',
    #  'keep edmTriggerResults_TriggerResults__REDIGI',
    #  'keep edmTriggerResults_TriggerResults__REDIGI36X'
    #)
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('START36_V8::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

# HLT filter
import HLTrigger.HLTfilters.hltHighLevel_cfi
process.EWK_MuHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
process.EWK_MuHLTFilter.throw = cms.bool(False)
process.EWK_MuHLTFilter.HLTPaths = ["HLT_Mu9","HLT_Mu11","HLT_Mu15"]

### ZMuMu candidates

# Muon candidates for Zmumu
process.goodMuonsForZ = cms.EDFilter("MuonSelector",
  src = cms.InputTag("muons"),
  cut = cms.string('pt > 20 && abs(eta)<2.4 && isGlobalMuon = 1 && isTrackerMuon = 1 && isolationR03().sumPt<3.0 && abs(innerTrack().dxy)<1.0'),
  filter = cms.bool(True)                                
)

# Z->mumu candidates
process.dimuons = cms.EDProducer("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(True),
    cut = cms.string('mass > 60'),
    decay = cms.string("goodMuonsForZ@+ goodMuonsForZ@-")
)

# Z filters
process.dimuonsFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("dimuons"),
    minNumber = cms.uint32(1)
)

# Z Skim path
process.EWK_dimuonsPath = cms.Path(
    process.EWK_MuHLTFilter *
    process.goodMuonsForZ *
    process.dimuons *
    process.dimuonsFilter
)

### Add WMuNu candidates

# Muons for Ws
process.goodMuonsForW = cms.EDFilter("MuonViewRefSelector",
  src = cms.InputTag("muons"),
  cut = cms.string('isGlobalMuon=1 && isTrackerMuon=1 && abs(eta)<2.1 && abs(globalTrack().dxy)<0.2 && pt>20. && globalTrack().normalizedChi2<10 && globalTrack().hitPattern().numberOfValidTrackerHits>10 && globalTrack().hitPattern().numberOfValidMuonHits>0 && globalTrack().hitPattern().numberOfValidPixelHits>0 && numberOfMatches>1 && (isolationR03().sumPt+isolationR03().emEt+isolationR03().hadEt)<0.15*pt'),
  filter = cms.bool(True)
)

# Cuts on Wmunu systems
process.wmnPFCands = cms.EDProducer("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(False),
    cut = cms.string('sqrt((daughter(0).pt+daughter(1).pt)*(daughter(0).pt+daughter(1).pt)-pt*pt)>50'),
    decay = cms.string("goodMuonsForW pfMet")
)
process.wmnPFFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("wmnPFCands"),
    minNumber = cms.uint32(1)
)

process.wmnTCCands = cms.EDProducer("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(False),
    cut = cms.string('sqrt((daughter(0).pt+daughter(1).pt)*(daughter(0).pt+daughter(1).pt)-pt*pt)>50'),
    decay = cms.string("goodMuonsForW tcMet")
)
process.wmnTCFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("wmnTCCands"),
    minNumber = cms.uint32(1)
)


# W Skim paths
process.EWK_pfMetWMuNusPath = cms.Path(
    process.EWK_MuHLTFilter *
    process.goodMuonsForW *
    process.wmnPFCands *
    process.wmnPFFilter
)

process.EWK_tcMetWMuNusPath = cms.Path(
    process.EWK_MuHLTFilter *
    process.goodMuonsForW *
    process.wmnTCCands *
    process.wmnTCFilter
)

# Output module configuration
from Configuration.EventContent.EventContent_cff import *
EWK_WZSkimEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
EWK_WZSkimEventContent.outputCommands.extend(FEVTEventContent.outputCommands)

EWK_WZSkimEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring(
           'EWK_dimuonsPath',
           'EWK_pfMetWMuNusPath',
           'EWK_tcMetWMuNusPath')
    )
)

process.EWK_WZSkimOutputModule = cms.OutputModule("PoolOutputModule",
    EWK_WZSkimEventContent,
    EWK_WZSkimEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('EWKWZSkim'),
        dataTier = cms.untracked.string('USER')
   ),
   fileName = cms.untracked.string('EWK_WZSkim_SD_Mu.root')
)

process.outpath = cms.EndPath(process.EWK_WZSkimOutputModule)


