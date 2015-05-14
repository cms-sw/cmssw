import FWCore.ParameterSet.Config as cms

HLTPath = "HLT_Ele*"
HLTProcessName = "HLT"

##    ____      __ _____ _           _                   
##   / ___|___ / _| ____| | ___  ___| |_ _ __ ___  _ __  
##  | |  _/ __| |_|  _| | |/ _ \/ __| __| '__/ _ \| '_ \ 
##  | |_| \__ \  _| |___| |  __/ (__| |_| | | (_) | | | |
##   \____|___/_| |_____|_|\___|\___|\__|_|  \___/|_| |_|
##  
#  GsfElectron ################ 

GsfMatchedPhotonCands = cms.EDProducer("ElectronMatchedCandidateProducer",
   src     = cms.InputTag("goodPhotons"),
   ReferenceElectronCollection = cms.untracked.InputTag("goodElectrons"),
   deltaR =  cms.untracked.double(0.3)
)

##    _____ _           _                     ___    _ 
##   | ____| | ___  ___| |_ _ __ ___  _ __   |_ _|__| |
##   |  _| | |/ _ \/ __| __| '__/ _ \| '_ \   | |/ _` |
##   | |___| |  __/ (__| |_| | | (_) | | | |  | | (_| |
##   |_____|_|\___|\___|\__|_|  \___/|_| |_| |___\__,_|
##   
# Electron ID  ######


                         
##    _____     _                         __  __       _       _     _             
##   |_   _| __(_) __ _  __ _  ___ _ __  |  \/  | __ _| |_ ___| |__ (_)_ __   __ _ 
##     | || '__| |/ _` |/ _` |/ _ \ '__| | |\/| |/ _` | __/ __| '_ \| | '_ \ / _` |
##     | || |  | | (_| | (_| |  __/ |    | |  | | (_| | || (__| | | | | | | | (_| |
##     |_||_|  |_|\__, |\__, |\___|_|    |_|  |_|\__,_|\__\___|_| |_|_|_| |_|\__, |
##                |___/ |___/                                                |___/ 
##   
# Trigger  ##################
PassingHLT = cms.EDProducer("trgMatchGsfElectronProducer",    
    InputProducer = cms.InputTag( 'gedGsfElectrons' ),                          
    hltTags = cms.untracked.string( HLTPath ),
    triggerEventTag = cms.untracked.InputTag("hltTriggerSummaryAOD","",HLTProcessName),
    triggerResultsTag = cms.untracked.InputTag("TriggerResults","",HLTProcessName)   
)

# WElecTagHLT = PassingHLT.clone(
#     InputProducer = cms.InputTag( "PassingWP80" )
#     )

# ele_sequence = cms.Sequence(
#     goodElectrons +
#     PassingWP80 +
#     WElecTagHLT
#     )


import HLTrigger.HLTfilters.hltHighLevel_cfi
ZSCHltFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    throw = cms.bool(False),
    HLTPaths = ['HLT_Ele27_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele15_CaloIdT_CaloIsoVL_trackless_v*']
    )

# elecMetSeq = cms.Sequence( WEnuHltFilter * ele_sequence * elecMetFilter )


selectedElectrons = cms.EDFilter("GsfElectronRefSelector",
                                 src = cms.InputTag( 'gedGsfElectrons' ),
                                 cut = cms.string(
#    "(abs(superCluster.eta)<2.5) && (energy*sin(superClusterPosition.theta)> 15)")
    "(abs(superCluster.eta)<3) && (energy*sin(superClusterPosition.theta)> 15)")
                                         )

selectedMuons = cms.EDFilter("MuonRefSelector",
                                 src = cms.InputTag( 'muons' ),
                                 cut = cms.string("")
                                         )

selectedPhotons = cms.EDFilter("PhotonRefSelector",
                                 src = cms.InputTag( 'gedPhotons' ),
                                 cut = cms.string(
    "(abs(superCluster.eta)<3) && (pt> 10)")
                                         )

# This are the cuts at trigger level except ecalIso
PassingVeryLooseId = selectedElectrons.clone(
    cut = cms.string(
    selectedElectrons.cut.value() +
    " && (gsfTrack.trackerExpectedHitsInner.numberOfHits<=1)" #wrt std WP90 allowing 1 numberOfMissingExpectedHits
    " && ((isEB"
    " && ( dr03TkSumPt/p4.Pt <0.2 "#&& dr03EcalRecHitSumEt/p4.Pt < 0.3
    " && dr03HcalTowerSumEt/p4.Pt  < 0.2 )"
    " && (sigmaIetaIeta<0.01)"
    " && ( -0.15<deltaPhiSuperClusterTrackAtVtx<0.15 )"
    " && ( -0.007<deltaEtaSuperClusterTrackAtVtx<0.007 )"
    " && (hadronicOverEm<0.12)"
    ")"
    " || (isEE"
    " && ( dr03TkSumPt/p4.Pt <0.2"
    #&& dr03EcalRecHitSumEt/p4.Pt < 0.3
    " && dr03HcalTowerSumEt/p4.Pt  < 0.2 )"
    " && (sigmaIetaIeta<0.03)"
    " && ( -0.10<deltaPhiSuperClusterTrackAtVtx<0.10 )"
    " && ( -0.009<deltaEtaSuperClusterTrackAtVtx<0.009 )"
    " && (hadronicOverEm<0.10) "
    "))"
    )
    )
PassingMediumId = selectedElectrons.clone(
    cut = cms.string(
    selectedElectrons.cut.value() +
    " && (gsfTrack.trackerExpectedHitsInner.numberOfHits<=1)" #wrt std WP90 allowing 1 numberOfMissingExpectedHits
    " && ((isEB"
    " && ( dr03TkSumPt/p4.Pt <0.2 "#&& dr03EcalRecHitSumEt/p4.Pt < 0.3
    " && dr03HcalTowerSumEt/p4.Pt  < 0.2 )"
    " && (sigmaIetaIeta<0.01)"
    " && ( -0.06<deltaPhiSuperClusterTrackAtVtx<0.06 )"
    " && ( -0.004<deltaEtaSuperClusterTrackAtVtx<0.004 )"
    " && (hadronicOverEm<0.12)"
    ")"
    " || (isEE"
    " && ( dr03TkSumPt/p4.Pt <0.2"
    #&& dr03EcalRecHitSumEt/p4.Pt < 0.3
    " && dr03HcalTowerSumEt/p4.Pt  < 0.2 )"
    " && (sigmaIetaIeta<0.03)"
    " && ( -0.03<deltaPhiSuperClusterTrackAtVtx<0.03 )"
    " && ( -0.007<deltaEtaSuperClusterTrackAtVtx<0.007 )"
    " && (hadronicOverEm<0.10) "
    "))"
    )
    )

PassingTightId = selectedElectrons.clone(
    cut = cms.string(
    selectedElectrons.cut.value() +
    " && (gsfTrack.trackerExpectedHitsInner.numberOfHits<=0)" #wrt std WP90 allowing 1 numberOfMissingExpectedHits
    " && ((isEB"
    " && ( dr03TkSumPt/p4.Pt <0.2 "#&& dr03EcalRecHitSumEt/p4.Pt < 0.3
    " && dr03HcalTowerSumEt/p4.Pt  < 0.2 )"
    " && (sigmaIetaIeta<0.01)"
    " && ( -0.03<deltaPhiSuperClusterTrackAtVtx<0.03 )"
    " && ( -0.004<deltaEtaSuperClusterTrackAtVtx<0.004 )"
    " && (hadronicOverEm<0.12)"
    ")"
    " || (isEE"
    " && ( dr03TkSumPt/p4.Pt <0.2"
    #&& dr03EcalRecHitSumEt/p4.Pt < 0.3
    " && dr03HcalTowerSumEt/p4.Pt  < 0.2 )"
    " && (sigmaIetaIeta<0.03)"
    " && ( -0.02<deltaPhiSuperClusterTrackAtVtx<0.02 )"
    " && ( -0.007<deltaEtaSuperClusterTrackAtVtx<0.007 )"
    " && (hadronicOverEm<0.10) "
    "))"
    )
    )

PassingMuonVeryLooseId = selectedMuons.clone(
    cut = cms.string(
    selectedMuons.cut.value() +
    "(isPFMuon) && (isGlobalMuon || isTrackerMuon)"
    )
    )

PassingPhotonVeryLooseId = selectedPhotons.clone(
    cut = cms.string(
    selectedPhotons.cut.value() +
    "&& ( (eta<1.479 && sigmaIetaIeta<0.02 && hadronicOverEm<0.06 )"
    "||"
    "( eta>=1.479 && sigmaIetaIeta<0.04 && hadronicOverEm<0.06 ) )"
    )
    )
    
MuFilter = cms.EDFilter("CandViewCountFilter",
                         src = cms.InputTag("PassingMuonVeryLooseId"),
                         minNumber = cms.uint32(2)
                         )
PhoFilter = cms.EDFilter("CandViewCountFilter",
                         src = cms.InputTag("PassingPhotonVeryLooseId"),
                         minNumber = cms.uint32(1)
                         ) 

#------------------------------ electronID producer
from Calibration.EleSelectionProducers.eleselectionproducers_cfi import *
# process.EleSelectionProducers

SCselector = cms.EDFilter("SuperClusterSelector",
                          src = cms.InputTag('correctedMulti5x5SuperClustersWithPreshower'),
                          cut = cms.string('(eta>2.4 || eta<-2.4) && (energy*sin(position.theta)> 15)')
                          )

### Build candidates from all the merged superclusters
eleSC = cms.EDProducer('ConcreteEcalCandidateProducer',
                  src = cms.InputTag('SCselector'),
                  particleType = cms.string('gamma')
                  )

selectedCands = cms.EDFilter("AssociatedVariableMaxCutCandRefSelector",
                             src = cms.InputTag("eleSelectionProducers:loose"),
                             max = cms.double("0.5")
                             )

eleSelSeq = cms.Sequence( selectedElectrons + PassingVeryLooseId + PassingTightId + 
                          (SCselector*eleSC)
                          )

muSelSeq = cms.Sequence( selectedMuons + PassingMuonVeryLooseId + PassingPhotonVeryLooseId + MuFilter + PhoFilter +
                          (SCselector*eleSC)
                          )

############################################################
# Selectors
##############################
ZeeSelector =  cms.EDProducer("CandViewShallowCloneCombiner",
                              decay = cms.string("PassingVeryLooseId PassingVeryLooseId"),
                              checkCharge = cms.bool(False),
                              cut   = cms.string("mass > 40 && mass < 140")
                              )


#met, mt cuts for W selection
MT="sqrt(2*daughter(0).pt*daughter(1).pt*(1 - cos(daughter(0).phi - daughter(1).phi)))"
MET_CUT_MIN = 25.
W_ELECTRON_ET_CUT_MIN = 30.0
MT_CUT_MIN = 50.

WenuSelector = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("pfMet PassingTightId"), # charge coniugate states are implied
    checkCharge = cms.bool(False),                           
    cut   = cms.string(("daughter(0).pt > %f && daughter(1).pt > %f && "+MT+" > %f") % (MET_CUT_MIN, W_ELECTRON_ET_CUT_MIN, MT_CUT_MIN))
)


EleSCSelector = cms.EDProducer("CandViewShallowCloneCombiner",
                               decay = cms.string("PassingVeryLooseId eleSC"),
#                               decay = cms.string("selectedElectrons eleSC"),
#                               decay = cms.string("PassingVeryLooseId photons"),
                               checkCharge = cms.bool(False), 
#                               cut   = cms.string("40 <mass < 140 && daughter(0).pt>27")
                               cut = cms.string("40 < mass < 140 ")
                               )
WZSelector = cms.EDProducer("CandViewMerger",
                            src = cms.VInputTag("WenuSelector", "ZeeSelector", "EleSCSelector")
                            )

 

############################################################
# Filters
##############################
WenuFilter = cms.EDFilter("CandViewCountFilter",
                          src = cms.InputTag("WenuSelector"),
                          minNumber = cms.uint32(1)
                          )
# filter events with at least one Zee candidate as identified by the ZeeSelector
ZeeFilter = cms.EDFilter("CandViewCountFilter",
                         src = cms.InputTag("ZeeSelector"),
                         minNumber = cms.uint32(1)
                         )

MinZSCNumberFilter = cms.EDFilter("CandViewCountFilter",
                                  #src = cms.InputTag('SCselector'),
                                  src = cms.InputTag('eleSC'),
                                  minNumber = cms.uint32(1)
                                  )
ZSCFilter = cms.EDFilter("CandViewCountFilter",
                         src = cms.InputTag("EleSCSelector"),
                         minNumber = cms.uint32(1)
                         )

WZFilter = cms.EDFilter("CandViewCountFilter",
                        src = cms.InputTag("WZSelector"),
                        minNumber = cms.uint32(1)
                        )

FilterSeq = cms.Sequence(eleSelSeq * (ZeeSelector + WenuSelector + EleSCSelector) * WZSelector)
ZSCSingleEleFilterSeq = cms.Sequence(~ZSCHltFilter * eleSelSeq * EleSCSelector * ZSCFilter)
FilterMuSeq = cms.Sequence(muSelSeq * (ZeeSelector + WenuSelector + EleSCSelector) * WZSelector)
