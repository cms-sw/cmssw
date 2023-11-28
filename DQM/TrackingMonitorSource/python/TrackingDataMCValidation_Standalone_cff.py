import FWCore.ParameterSet.Config as cms
from DQM.TrackingMonitorSource.StandaloneTrackMonitor_cfi import *
from DQM.TrackingMonitorSource.ZEEDetails_cfi import *
# from DQM.TrackingMonitor.V0Monitor_cfi import *

# Primary Vertex Selector
selectedPrimaryVertices = cms.EDFilter("VertexSelector",
                                       src = cms.InputTag('offlinePrimaryVertices'),
                                      # cut = cms.string("!isFake && ndof >= 4 && abs(z) < 24 && abs(position.Rho) < 2.0"),
                                       cut = cms.string(""),
                                       filter = cms.bool(True)
                                       )
# Track Selector
selectedTracks = cms.EDFilter("TrackSelector",
                              src = cms.InputTag('generalTracks'),
                              cut = cms.string("pt > 1.0"),
                              #cut = cms.string(""),
                              filter = cms.bool(True)
                              )

# Track Multiplicity Selector
selectedMultiplicityTracks = cms.EDFilter("TrackMultiplicityFilter",
                                          src = cms.InputTag('generalTracks'),
                                          #cut = cms.string("pt > 1.0"),
                                          nmin = cms.untracked.uint32(500),
                                          cut = cms.untracked.string(""),
                                          filter = cms.bool(True)
                                      )

# Track ALCARECO Selection for zerobias
selectedAlcaRecoZBTracks = cms.EDProducer("AlcaRecoTrackSelector",
                                        src = cms.InputTag('generalTracks'),
                                        #cut = cms.string("pt > 0.65 && abs(eta) < 3.5 && p > 1.5 && hitPattern.numberOfAllHits('TRACK_HITS') > 7"),
                                        #cut = cms.string(""),
                                          ptmin = cms.untracked.double(0.65),
                                        pmin = cms.untracked.double(1.5),
                                        etamin = cms.untracked.double(-3.5),
                                        etamax = cms.untracked.double(3.5),
                                          nhits = cms.untracked.uint32(7)
)
'''
# Track ALCARECO Selection for singlemuon
selectedAlcaRecoSMTracks = cms.EDFilter("TrackSelector",
                              src = cms.InputTag('selectedMultiplicityTracks'),
                              cut = cms.string("pt > 2.0 && abs(eta) < 3.5 && p > 1.5 && hitPattern.numberOfAllTrackerHits > 7"),
                              #cut = cms.string(""),
                              filter = cms.bool(True)
                              )
'''
# HLT path selector
hltPathFilter = cms.EDFilter("HLTPathSelector",
                             processName = cms.string("HLT"),
                             verbose = cms.untracked.bool(False),
                             hltPathsOfInterest = cms.vstring("HLT_ZeroBias_v"),
                             triggerResults = cms.untracked.InputTag("TriggerResults","","HLT"),
                             triggerEvent = cms.untracked.InputTag("hltTriggerSummaryAOD","","HLT")
                             )

# HLT path selector Muon
hltPathFilterMuon = cms.EDFilter("HLTPathSelector",
                             processName = cms.string("HLT"),
                             verbose = cms.untracked.bool(False),
                             hltPathsOfInterest = cms.vstring("HLT_IsoMu24_v"),
                             triggerResults = cms.untracked.InputTag("TriggerResults","","HLT"),
                             triggerEvent = cms.untracked.InputTag("hltTriggerSummaryAOD","","HLT")
                             )

# HLT path selector Electron
hltPathFilterElectron = cms.EDFilter("HLTPathSelector",
                             processName = cms.string("HLT"),
                             verbose = cms.untracked.bool(False),
                             hltPathsOfInterest = cms.vstring("HLT_Ele32_WPTight_Gsf_v"),
                             triggerResults = cms.untracked.InputTag("TriggerResults","","HLT"),
                             triggerEvent = cms.untracked.InputTag("hltTriggerSummaryAOD","","HLT")
                             )

# HLT path selector ttbar
hltPathFilterTtbar = cms.EDFilter("HLTPathSelector",
                             processName = cms.string("HLT"),
                             verbose = cms.untracked.bool(False),
                             hltPathsOfInterest = cms.vstring("HLT_Ele32_WPTight_Gsf_v","HLT_IsoMu24_v"),
                             triggerResults = cms.untracked.InputTag("TriggerResults","","HLT"),
                             triggerEvent = cms.untracked.InputTag("hltTriggerSummaryAOD","","HLT")
                             )

# Z->MuMu event selector
ztoMMEventSelector = cms.EDFilter("ZtoMMEventSelector")
muonTracks = cms.EDProducer("ZtoMMMuonTrackProducer")
# Z->ee event selector
ztoEEEventSelector = cms.EDFilter("ZtoEEEventSelector")
electronTracks = cms.EDProducer("ZtoEEElectronTrackProducer")
#ttbar event selector
ttbarEventSelector = cms.EDFilter("ttbarEventSelector")
ttbarTracks = cms.EDProducer("TtbarTrackProducer")

# Added module for V0Monitoring for Ks only
# KshortMonitor = v0Monitor.clone()
# KshortMonitor.FolderName = cms.string("Tracking/V0Monitoring/Ks")
# KshortMonitor.v0         = cms.InputTag('generalV0Candidates:Kshort')
# KshortMonitor.histoPSet.massPSet = cms.PSet(
#   nbins = cms.int32 ( 100 ),
#   xmin  = cms.double( 0.400),
#   xmax  = cms.double( 0.600),
# )

# For MinBias
standaloneTrackMonitorMC = standaloneTrackMonitor.clone(
    puScaleFactorFile = "PileupScaleFactor_316060_wrt_nVertex_ZeroBias.root",
    doPUCorrection    = True,
    isMC              = True
    )
standaloneValidationMinbias = cms.Sequence(
    hltPathFilter
    * selectedPrimaryVertices 
#    * selectedMultiplicityTracks  # Use selectedMultiplicityTracks if needed nTracks > desired multiplicity
#    * selectedAlcaRecoZBTracks
    * selectedTracks
    * standaloneTrackMonitor)
standaloneValidationMinbiasMC = cms.Sequence(
    hltPathFilter
    * selectedPrimaryVertices 
#    * selectedMultiplicityTracks  # Use selectedMultiplicityTracks if needed nTracks > desired multiplicity
#    * selectedAlcaRecoZBTracks
    * selectedTracks
    * standaloneTrackMonitorMC)
# For ZtoEE
standaloneTrackMonitorElec = standaloneTrackMonitor.clone(
    folderName = "ElectronTracks",
    trackInputTag = 'electronTracks',
    )

standaloneTrackMonitorElecMC = standaloneTrackMonitor.clone(
    folderName = "ElectronTracks",
    trackInputTag = 'electronTracks',
    puScaleFactorFile = "PileupScaleFactor_316082_wrt_nVertex_DYToLL.root",
    doPUCorrection    = True,
    isMC              = True
    )

ZEEDetailsMC = ZEEDetails.clone(
    puScaleFactorFile = "PileupScaleFactor_316082_wrt_nVertex_DYToLL.root",
    doPUCorrection    = True,
    isMC              = True
    )

standaloneValidationElec = cms.Sequence(
    hltPathFilterElectron
    * selectedTracks
    * selectedPrimaryVertices
    * ztoEEEventSelector
    * electronTracks
    * standaloneTrackMonitorElec   
    * standaloneTrackMonitor
    * ZEEDetails)
standaloneValidationElecMC = cms.Sequence(
    hltPathFilterElectron
    * selectedTracks
    * selectedPrimaryVertices
    * ztoEEEventSelector
    * electronTracks
    * standaloneTrackMonitorElecMC   
    * standaloneTrackMonitorMC
    * ZEEDetailsMC)
# For ZtoMM
standaloneTrackMonitorMuon = standaloneTrackMonitor.clone(
    folderName = "MuonTracks",
    trackInputTag = 'muonTracks',
    )
standaloneTrackMonitorMuonMC = standaloneTrackMonitor.clone(
    folderName = "MuonTracks",
    trackInputTag = 'muonTracks',
    puScaleFactorFile = "PileupScaleFactor_316082_wrt_nVertex_DYToLL.root",
    doPUCorrection    = True,
    isMC              = True
    )

standaloneValidationMuon = cms.Sequence(
    hltPathFilterMuon
    * selectedTracks
    * selectedPrimaryVertices
    * ztoMMEventSelector
    * muonTracks
    * standaloneTrackMonitorMuon
    * standaloneTrackMonitor)

standaloneValidationMuonMC = cms.Sequence(
    hltPathFilterMuon
    * selectedTracks
    * selectedPrimaryVertices
    * ztoMMEventSelector
    * muonTracks
    * standaloneTrackMonitorMuonMC 
    * standaloneTrackMonitorMC)

# For ttbar
standaloneTrackMonitorTTbar = standaloneTrackMonitor.clone(
    folderName = "TTbarTracks",
    trackInputTag = 'ttbarTracks',
    )

standaloneTrackMonitorTTbarMC = standaloneTrackMonitor.clone(
    folderName = "TTbarTracks",
    trackInputTag = 'ttbarTracks',
    puScaleFactorFile = "PileupScaleFactor_316082_wrt_nVertex_DYToLL.root",
    doPUCorrection    = True,
    isMC              = True
    )

standaloneValidationTTbar = cms.Sequence(
    hltPathFilterTtbar
    * selectedPrimaryVertices
    * ttbarEventSelector
    * ttbarTracks
#    * selectedTracks
    * standaloneTrackMonitorTTbar)

standaloneValidationTTbarMC = cms.Sequence(
    hltPathFilterTtbar
    * selectedPrimaryVertices
    * ttbarEventSelector
    * ttbarTracks
#    * selectedTracks
    * standaloneTrackMonitorTTbarMC)
