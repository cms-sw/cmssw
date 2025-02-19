##____________________________________________________________________________||
from PhysicsTools.PatAlgos.patTemplate_cfg import *

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing ('standard')

options.register('GlobalTag', "GR_R_52_V7::All", VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "GlobaTTag to use (otherwise default Pat GT is used)")
options.register('mcInfo', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "process MonteCarlo data, default is data")
options.register('jetCorrections', 'L2Relative', VarParsing.VarParsing.multiplicity.list, VarParsing.VarParsing.varType.string, "Level of jet corrections to use: Note the factors are read from DB via GlobalTag")
options.jetCorrections.append('L3Absolute')
options.register('doJetPFchs', True, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "process MonteCarlo data, default is data")

options.register('hltName', 'HLT', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "HLT menu to use for trigger matching, e.g., HLT, REDIGI311X")
options.register('mcVersion', '', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "'36X' for example. Used for specific MC fix")
options.register('jetTypes', 'AK5PF', VarParsing.VarParsing.multiplicity.list, VarParsing.VarParsing.varType.string, "Additional jet types that will be produced (AK5Calo and AK5PF, cross cleaned in PF2PAT, are included anyway)")
#options.jetTypes.append('AK5Calo')
options.register('hltSelection', '*', VarParsing.VarParsing.multiplicity.list, VarParsing.VarParsing.varType.string, "hlTriggers (OR) used to filter events. for data: ''HLT_Mu9', 'HLT_IsoMu9', 'HLT_IsoMu13_v*''; for MC, HLT_Mu9")
options.register('addKeep', '', VarParsing.VarParsing.multiplicity.list, VarParsing.VarParsing.varType.string, "Additional keep and drop statements to trim the event content")

options.register('dataVersion', '', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "'36X' for example. Used for specific DATA fix")

options.register('debug', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "switch on/off debug mode")

options.register('type', 'METScanning', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "key output type string")

options.register('dataTier', 'AOD', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "data tier string, e.g., AOD, RECO")

options.parseArguments()
options._tagOrder =[]

print options

#-- Message Logger ------------------------------------------------------------
process.MessageLogger.cerr.FwkReport.reportEvery = 1
if options.debug:
   process.MessageLogger.cerr.FwkReport.reportEvery = 1

#-- Input Source --------------------------------------------------------------
if options.files:
   process.source.fileNames = options.files
else:
   process.source.fileNames = [
'file:pickevents_merged.root'
   ]

process.source.inputCommands = cms.untracked.vstring( "keep *", "drop *_MEtoEDMConverter_*_*" )
process.maxEvents.input = options.maxEvents

# Calibration tag -----------------------------------------------------------
if options.GlobalTag:
   process.GlobalTag.globaltag = options.GlobalTag

# JEC
if options.mcInfo == False: options.jetCorrections.append('L2L3Residual')
options.jetCorrections.insert(0, 'L1FastJet')

##____________________________________________________________________________||
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))
process.out.fileName = cms.untracked.string('skim.root')

# load the PAT config
process.load("PhysicsTools.PatAlgos.patSequences_cff")

# Configure PAT to use PF2PAT instead of AOD sources
# this function will modify the PAT sequences. It is currently 
# not possible to run PF2PAT+PAT and standart PAT at the same time
from PhysicsTools.PatAlgos.tools.pfTools import *

# This is for PFCHS
from PhysicsTools.SelectorUtils.pvSelector_cfi import pvSelector
process.goodOfflinePrimaryVertices = cms.EDFilter(
    "PrimaryVertexObjectFilter",
    filterParams = pvSelector.clone( minNdof = cms.double(4.0), maxZ = cms.double(24.0) ),
    src=cms.InputTag('offlinePrimaryVertices')
)

postfix = "PFlow"
jetAlgo="AK5"
print "====> Configuring usePF2PAT : using AK5PFchs ..."
print "See https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookJetEnergyCorrections#JetEnCorPFnoPU"
usePF2PAT(process, runPF2PAT=True, jetAlgo=jetAlgo, runOnMC=options.mcInfo, postfix=postfix, jetCorrections=('AK5PFchs', options.jetCorrections))
getattr(process,"pfPileUp"+postfix).Enable = True
getattr(process,"pfPileUp"+postfix).Vertices = 'goodOfflinePrimaryVertices'
getattr(process,"pfPileUp"+postfix).checkClosestZVertex = cms.bool(False)
getattr(process,"pfJets"+postfix).doAreaFastjet = True
getattr(process,"pfJets"+postfix).doRhoFastjet = False
getattr(process,"patJetCorrFactors"+postfix).rho = cms.InputTag("kt6PFJets", "rho")

process.load('RecoJets.JetProducers.kt4PFJets_cfi')
process.kt6PFJets = process.kt4PFJets.clone( rParam = 0.6, doAreaFastjet = True, doRhoFastjet = True )

# top projections in PF2PAT:
getattr(process,"pfNoPileUp"+postfix).enable = True
getattr(process,"pfNoMuon"+postfix).enable = True
getattr(process,"pfNoElectron"+postfix).enable = True
getattr(process,"pfNoTau"+postfix).enable = False
getattr(process,"pfNoJet"+postfix).enable = True

# verbose flags for the PF2PAT modules
getattr(process,"pfNoMuon"+postfix).verbose = False

# Add the PV selector and KT6 producer to the sequence
getattr(process,"patPF2PATSequence"+postfix).replace(
    getattr(process,"pfNoElectron"+postfix),
    getattr(process,"pfNoElectron"+postfix)*process.kt6PFJets )

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
process.hcalLaserEventFilter.vetoByRunEventNumber=cms.untracked.bool(False)
process.hcalLaserEventFilter.vetoByHBHEOccupancy=cms.untracked.bool(True)

##____________________________________________________________________________||
process.load('RecoMET.METFilters.EcalDeadCellTriggerPrimitiveFilter_cfi')
process.EcalDeadCellTriggerPrimitiveFilter.debug = cms.bool(True)

##____________________________________________________________________________||
process.load('RecoMET.METFilters.EcalDeadCellBoundaryEnergyFilter_cfi')

process.load('RecoMET.METFilters.jetIDFailureFilter_cfi')
process.jetIDFailure.MinJetPt  = cms.double(30.0)
process.jetIDFailure.MaxJetEta = cms.double(999.0)

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
## Let it run
#process.pf2patDUMMY = cms.Path(
#    process.goodOfflinePrimaryVertices *
#    getattr(process,"patPF2PATSequence"+postfix)
#)

process.load('RecoMET.METFilters.eeNoiseFilter_cfi')

process.load('RecoMET/METAnalyzers/CSCHaloFilter_cfi')

process.rejectRecov = cms.EDFilter(
  "RecovRecHitFilter",
  EERecHitSource = cms.InputTag("reducedEcalRecHitsEE"),
  MinRecovE = cms.double(30),
  TaggingMode=cms.bool(False)
)

process.Vertex = cms.Path(process.goodOfflinePrimaryVertices*getattr(process,"patPF2PATSequence"+postfix)*~process.primaryVertexFilter)
process.Scraping = cms.Path(~process.noscraping)
process.HBHENoise = cms.Path(~process.HBHENoiseFilter)
process.CSCTightHalo = cms.Path(~process.CSCTightHaloFilter)
process.RecovRecHit = cms.Path(~process.rejectRecov)

process.HCALLaser = cms.Path(~process.hcalLaserEventFilter)
process.ECALDeadCellTP = cms.Path(~process.EcalDeadCellTriggerPrimitiveFilter)
process.ECALDeadCellBE = cms.Path(~process.EcalDeadCellBoundaryEnergyFilter)
process.jetID = cms.Path(~process.jetIDFailure)
process.trackingFailure = cms.Path(process.goodVertices*~process.trackingFailureFilter)
process.inconsistentMuon = cms.Path(~process.inconsistentMuonPFCandidateFilter)
process.greedyMuon = cms.Path(~process.greedyMuonPFCandidateFilter)
process.eeNoise = cms.Path(~process.eeNoiseFilter)

##____________________________________________________________________________||
process.hltTriggerSummaryAOD = cms.EDProducer(
    "TriggerSummaryProducerAOD",
    processName = cms.string( "@" )
    )

##____________________________________________________________________________||
from PhysicsTools.PatAlgos.tools.trigTools import *
switchOnTrigger(process)

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
