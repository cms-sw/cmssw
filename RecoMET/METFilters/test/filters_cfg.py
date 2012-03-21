##____________________________________________________________________________||
from PhysicsTools.PatAlgos.patTemplate_cfg import *

##____________________________________________________________________________||
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.out.fileName = cms.untracked.string('skim.root')
process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.GlobalTag.globaltag = 'GR_R_42_V19::All'

##____________________________________________________________________________||
process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
'file:/eos/uscms/store/user/seema/SusyRA2Analysis/17November2011_edmPickEvents_RA2FilterRejected/pickevents_05Aug2011_1_1_JQq.root',
'file:/eos/uscms/store/user/seema/SusyRA2Analysis/17November2011_edmPickEvents_RA2FilterRejected/pickevents_PromptRecoV4_1_1_g0v.root',
'file:/eos/uscms/store/user/seema/SusyRA2Analysis/17November2011_edmPickEvents_RA2FilterRejected/pickevents_PromptRecoV6_1_1_lEL.root',
'file:/eos/uscms/store/user/seema/SusyRA2Analysis/17November2011_edmPickEvents_RA2FilterRejected/pickevents_Run2011BPromptRecoV1_1_1_Gjg.root',
'file:/eos/uscms/store/user/seema/SusyRA2Analysis/17November2011_edmPickEvents_RA2FilterRejected/pickevents_Run2011BPromptRecoV1_2_1_sTu.root',
'file:/eos/uscms/store/user/seema/SusyRA2Analysis/17November2011_edmPickEvents_RA2FilterRejected/pickevents_may10ReReco_1_1_lcX.root',
        )
    )

##____________________________________________________________________________||
process.primaryVertexFilter = cms.EDFilter(
    "VertexSelector",
    src = cms.InputTag("offlinePrimaryVertices"),
    cut = cms.string("!isFake && ndof > 4 && abs(z) <= 24 && position.Rho <= 2"),
    filter = cms.bool(True)
    )

##____________________________________________________________________________||
process.noscraping = cms.EDFilter(
    "FilterOutScraping",
    applyfilter = cms.untracked.bool(True),
    debugOn = cms.untracked.bool(False),
    numtrack = cms.untracked.uint32(10),
    thresh = cms.untracked.double(0.25)
    )

##____________________________________________________________________________||
process.load('CommonTools/RecoAlgos/HBHENoiseFilter_cfi')
process.HBHENoiseFilter.minIsolatedNoiseSumE = cms.double(999999.)
process.HBHENoiseFilter.minNumIsolatedNoiseChannels = cms.int32(999999)
process.HBHENoiseFilter.minIsolatedNoiseSumEt = cms.double(999999.)

##____________________________________________________________________________||
#process.load('RecoMET.METAnalyzers.CSCHaloFilter_cfi')

##____________________________________________________________________________||
process.load("RecoMET.METFilters.hcalLaserEventFilter_cfi")
#process.hcalLaserEventFilter.vetoByRunEventNumber=cms.untracked.bool(False)
#process.hcalLaserEventFilter.vetoByHBHEOccupancy=cms.untracked.bool(True)

##____________________________________________________________________________||
process.load('RecoMET.METFilters.EcalDeadCellTriggerPrimitiveFilter_cfi')

##____________________________________________________________________________||
process.load('RecoMET.METFilters.EcalDeadCellBoundaryEnergyFilter_cfi')

process.goodVertices = cms.EDFilter(
  "VertexSelector",
  filter = cms.bool(False),
  src = cms.InputTag("offlinePrimaryVertices"),
  cut = cms.string("!isFake && ndof > 4 && abs(z) <= 24 && position.rho < 2")
)

process.load('RecoMET.METFilters.trackingFailureFilter_cfi')
process.load('JetMETCorrections.Configuration.DefaultJEC_cff')
#process.trackingFailureFilter.JetSource = cms.InputTag('ak5PFJets')
#process.trackingFailureFilter.JetSource = cms.InputTag('ak5PFJetsL2L3Residual')

process.load('RecoMET.METFilters.inconsistentMuonPFCandidateFilter_cfi')

process.load('RecoMET.METFilters.greedyMuonPFCandidateFilter_cfi')

##____________________________________________________________________________||
#process.RecovRecHitFilter = cms.EDFilter(
#  "RecovRecHitFilter",
#  EERecHitSource = cms.InputTag("ecalRecHit:EcalRecHitsEE"),
#  MinRecovE = cms.double(30),
#  TaggingMode = cms.bool(False)
#)

##____________________________________________________________________________||
process.Vertex = cms.Path(~process.primaryVertexFilter)
process.Scraping = cms.Path(~process.noscraping)
#process.HBHENoise = cms.Path(~process.HBHENoiseFilter)
#process.CSCTightHalo = cms.Path(~process.CSCTightHaloFilter)
#process.RecovRecHit = cms.Path(process.RecovRecHitFilter)

process.HCALLaser = cms.Path(~process.hcalLaserEventFilter)
process.ECALDeadCellTP = cms.Path(~process.EcalDeadCellTriggerPrimitiveFilter)
process.ECALDeadCellBE = cms.Path(~process.EcalDeadCellBoundaryEnergyFilter)
process.trackingFailure = cms.Path(process.goodVertices*~process.trackingFailureFilter)
process.inconsistentMuon = cms.Path(~process.inconsistentMuonPFCandidateFilter)
process.greedyMuon = cms.Path(~process.greedyMuonPFCandidateFilter)

##____________________________________________________________________________||
process.hltTriggerSummaryAOD = cms.EDProducer(
    "TriggerSummaryProducerAOD",
    processName = cms.string( "@" )
    )

##____________________________________________________________________________||
from PhysicsTools.PatAlgos.tools.trigTools import *
switchOnTrigger(process)


##____________________________________________________________________________||
process.load("PhysicsTools.PatAlgos.patSequences_cff")

##____________________________________________________________________________||
process.patTriggerFilter = process.patTrigger.clone()
process.patTriggerFilter.processName = cms.string('PAT')
process.MessageLogger.suppressWarning += ['patTriggerFilter']

process.outpath = cms.EndPath(
    process.patTrigger *
    process.hltTriggerSummaryAOD *
    process.patTriggerFilter *
    process.out
    )

##____________________________________________________________________________||
from PhysicsTools.PatAlgos.patEventContent_cff import *
process.out.outputCommands = cms.untracked.vstring(
    'drop *',
    'keep patTriggerPaths_patTrigger*_*_*',
    ) 
process.out.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('*', '!*'))

##____________________________________________________________________________||
