import FWCore.ParameterSet.Config as cms

#HLTPath = "HLT_Ele*"
#HLTProcessName = "HLT"
myEleCollection =  cms.InputTag("gedGsfElectrons")

MinEleNumberFilter = cms.EDFilter("CandViewCountFilter",
                                  src = myEleCollection,
                                  minNumber = cms.uint32(1)
                                  )

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



                         
##    _____     _                         __  __       _       _     _             
##   |_   _| __(_) __ _  __ _  ___ _ __  |  \/  | __ _| |_ ___| |__ (_)_ __   __ _ 
##     | || '__| |/ _` |/ _` |/ _ \ '__| | |\/| |/ _` | __/ __| '_ \| | '_ \ / _` |
##     | || |  | | (_| | (_| |  __/ |    | |  | | (_| | || (__| | | | | | | | (_| |
##     |_||_|  |_|\__, |\__, |\___|_|    |_|  |_|\__,_|\__\___|_| |_|_|_| |_|\__, |
##                |___/ |___/                                                |___/ 
##   
# Trigger  ##################
#PassingHLT = cms.EDProducer("trgMatchGsfElectronProducer",    
#    InputProducer = myEleCollection,
#    hltTags = cms.untracked.string( HLTPath ),
#    triggerEventTag = cms.untracked.InputTag("hltTriggerSummaryAOD","",HLTProcessName),
#    triggerResultsTag = cms.untracked.InputTag("TriggerResults","",HLTProcessName)   
#)



import HLTrigger.HLTfilters.hltHighLevel_cfi
ZSCHltFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    throw = cms.bool(False),
    HLTPaths = ['HLT_Ele27_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele15_CaloIdT_CaloIsoVL_trackless_v*']
    )

# elecMetSeq = cms.Sequence( WEnuHltFilter * ele_sequence * elecMetFilter )

###MC check efficiency
genEleFromZ = cms.EDFilter("CandViewSelector",
                              src = cms.InputTag("genParticles"),
                              cut = cms.string("(obj.pdgId() == 11 || obj.pdgId() == -11) && obj.eta() <2.5 && obj.eta() > -2.5 && obj.pt() > 15 && obj.mother(0)->pdgId() == 23"),
                              )

genEleFromW = cms.EDFilter("CandViewSelector",
                              src = cms.InputTag("genParticles"),
                              cut = cms.string("(obj.pdgId() == 11 || obj.pdgId() == -11) && obj.eta() <2.5 && obj.eta() > -2.5 && obj.pt() > 25 && (obj.mother(0)->pdgId() == 24 || obj.mother(0)->pdgId() == -24)"),
                              )

genNuFromW = cms.EDFilter("CandViewSelector",
                              src = cms.InputTag("genParticles"),
                              cut = cms.string("(obj.pdgId() == 12 || obj.pdgId() == -12) && (obj.mother(0)->pdgId() == 24 || obj.mother(0)->pdgId() == -24)"),
                              )

combZ = cms.EDProducer("CandViewShallowCloneCombiner",
                                 decay = cms.string("genEleFromZ genEleFromZ"),
                                 checkCharge = cms.bool(False),
                                 cut = cms.string("40 < obj.mass() && obj.mass() < 140"),
                                 )

combW = cms.EDProducer("CandViewShallowCloneCombiner",
                                 decay = cms.string("genEleFromW genNuFromW"),
                                 checkCharge = cms.bool(False),
                                 cut = cms.string(""),
                                 )

ZFilterMC = cms.EDFilter("CandViewCountFilter",
                         src = cms.InputTag("combZ"),
                         minNumber = cms.uint32(1)
                         )

WFilterMC = cms.EDFilter("CandViewCountFilter",
                         src = cms.InputTag("combW"),
                         minNumber = cms.uint32(1)
                         )


##    _____ _           _                     ___    _ 
##   | ____| | ___  ___| |_ _ __ ___  _ __   |_ _|__| |
##   |  _| | |/ _ \/ __| __| '__/ _ \| '_ \   | |/ _` |
##   | |___| |  __/ (__| |_| | | (_) | | | |  | | (_| |
##   |_____|_|\___|\___|\__|_|  \___/|_| |_| |___\__,_|
##   
# Electron ID  ######

selectedECALElectrons = cms.EDFilter("GsfElectronRefSelector",
                                 src = myEleCollection,
                                 cut = cms.string(
    "(std::abs(obj.superCluster()->eta())<3) && (obj.energy()*std::sin(obj.superClusterPosition().theta())> 15)")
                                         )

selectedECALMuons = cms.EDFilter("MuonRefSelector",
                                 src = cms.InputTag( 'muons' ),
                                 cut = cms.string("")
                                         )

selectedECALPhotons = cms.EDFilter("PhotonRefSelector",
                                 src = cms.InputTag( 'gedPhotons' ),
                                 cut = cms.string(
    "(std::abs(obj.superCluster()->eta)<3) && (obj.pt() > 10)")
                                         )


# This are the cuts at trigger level except ecalIso
PassingVetoId = selectedECALElectrons.clone(
    cut = cms.string(
    selectedECALElectrons.cut.value() +
    " && (obj.gsfTrack()->hitPattern().numberOfHits(reco::HitPattern::MISSING_INNER_HITS)<=2)"
    " && ((obj.isEB()"
    " && ( ((obj.pfIsolationVariables().sumChargedHadronPt + std::max(0.0,obj.pfIsolationVariables().sumNeutralHadronEt + obj.pfIsolationVariables().sumPhotonEt - 0.5 * obj.pfIsolationVariables().sumPUPt))/obj.p4().pt())<0.164369)"
    " && (obj.full5x5_sigmaIetaIeta()<0.011100)"
    " && ( std::abs(obj.deltaPhiSuperClusterTrackAtVtx())  < 	0.252044 )"
    " && ( std::abs(obj.deltaEtaSuperClusterTrackAtVtx())  <  0.016315 )"
    " && (obj.hadronicOverEm()<0.345843)"
    ")"
    " || (obj.isEE()"
    " && (obj.gsfTrack()->hitPattern().numberOfHits(reco::HitPattern::MISSING_INNER_HITS)<=3)"
    " && ( ((obj.pfIsolationVariables().sumChargedHadronPt + std::max(0.0,obj.pfIsolationVariables().sumNeutralHadronEt + obj.pfIsolationVariables().sumPhotonEt - 0.5 * obj.pfIsolationVariables().sumPUPt))/obj.p4().pt())<0.212604 )"
    " && (obj.full5x5_sigmaIetaIeta()<0.033987)"
    " && ( std::abs(obj.deltaPhiSuperClusterTrackAtVtx())<0.245263 )"
    " && ( std::abs(obj.deltaEtaSuperClusterTrackAtVtx())<0.010671 )"
    " && (obj.hadronicOverEm()<0.134691) "
    "))"
    )
    )

PassingMuonVeryLooseId = selectedECALMuons.clone(
    cut = cms.string(
    selectedECALMuons.cut.value() +
    "(obj.isPFMuon()) && (obj.isGlobalMuon() || obj.isTrackerMuon())"
    )
    )

PassingPhotonVeryLooseId = selectedECALPhotons.clone(
    cut = cms.string(
    selectedECALPhotons.cut.value() +
    "&& ( (obj.eta()<1.479 && obj.sigmaIetaIeta()<0.02 && obj.hadronicOverEm()<0.06 )"
    "||"
    "( obj.eta()>=1.479 && obj.sigmaIetaIeta()<0.04 && obj.hadronicOverEm()<0.06 ) )"
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
SCselector = cms.EDFilter("SuperClusterSelector",
                          src = cms.InputTag('correctedMulti5x5SuperClustersWithPreshower'),
                          cut = cms.string('(obj.eta()>2.4 || obj.eta()<-2.4) && (obj.energy()*std::sin(obj.position().theta())> 15)')
                          )

### Build candidates from all the merged superclusters
eleSC = cms.EDProducer('ConcreteEcalCandidateProducer',
                  src = cms.InputTag('SCselector'),
                  particleType = cms.string('gamma')
                  )

# selectedCands = cms.EDFilter("AssociatedVariableMaxCutCandRefSelector",
#                              src = cms.InputTag("eleSelectionProducers:loose"),
#                              max = cms.double("0.5")
#                              )

#ecalCandidateMerged =  cms.EDProducer("CandViewMerger",
#                                      src = cms.VInputTag("PassingVetoId", "eleSC")
#                                      )

eleSelSeq = cms.Sequence( selectedECALElectrons + PassingVetoId +
                          (SCselector*eleSC)
                          )

muSelSeq = cms.Sequence( selectedECALMuons + selectedECALPhotons + PassingMuonVeryLooseId + PassingPhotonVeryLooseId + MuFilter + PhoFilter +
                          (SCselector*eleSC)
                          )


############################################################
# Selectors
##############################
ZeeSelector =  cms.EDProducer("CandViewShallowCloneCombiner",
                              decay = cms.string("PassingVetoId PassingVetoId"),
                              checkCharge = cms.bool(False),
                              cut   = cms.string("40 < obj.mass() && obj.mass() < 140")
                              )


#met, mt cuts for W selection
MT="std::sqrt(2*obj.daughter(0)->pt()*obj.daughter(1)->pt()*(1 - std::cos(obj.daughter(0)->phi() - obj.daughter(1)->phi())))"
MET_CUT_MIN = 25.
W_ELECTRON_ET_CUT_MIN = 30.0
MT_CUT_MIN = 50.

WenuSelector = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("pfMet PassingVetoId"), # charge coniugate states are implied
    checkCharge = cms.bool(False),                           
    cut   = cms.string(("obj.daughter(0)->pt() > %f && obj.daughter(1)->pt() > %f && "+MT+" > %f") % (MET_CUT_MIN, W_ELECTRON_ET_CUT_MIN, MT_CUT_MIN))
)


EleSCSelector = cms.EDProducer("CandViewShallowCloneCombiner",
                               decay = cms.string("PassingVetoId eleSC"),
                               checkCharge = cms.bool(False), 
                               cut = cms.string("40 < obj.mass() && obj.mass() < 140")
                               )

# for filtering events passing at least one of the filters
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

ZSCFilter = cms.EDFilter("CandViewCountFilter",
                         src = cms.InputTag("EleSCSelector"),
                         minNumber = cms.uint32(1)
                         )

# filter for events passing at least one of the other filters
WZFilter = cms.EDFilter("CandViewCountFilter",
                        src = cms.InputTag("WZSelector"),
                        minNumber = cms.uint32(1)
                        )



############################################################
# Sequences
##############################

preFilterSeq = cms.Sequence(MinEleNumberFilter)

selectorProducerSeq = cms.Sequence(eleSelSeq * (ZeeSelector + WenuSelector + EleSCSelector) * WZSelector)

ZeeSkimFilterSeq = cms.Sequence(preFilterSeq * selectorProducerSeq * 
                                 ZeeFilter)
ZSCSkimFilterSeq = cms.Sequence(preFilterSeq * selectorProducerSeq * 
                                 ~ZeeFilter * ZSCFilter)
WenuSkimFilterSeq = cms.Sequence(preFilterSeq * selectorProducerSeq * 
                                 ~ZeeFilter * ~ZSCFilter * WenuFilter)


checkMCZSeq = cms.Sequence(genEleFromZ * combZ * ZFilterMC) #sequence to check Zskim efficiency respect to the MC
checkMCWSeq = cms.Sequence(genEleFromW * genNuFromW * combW * WFilterMC) #sequence to check Wskim efficiency respect to the MC

#FilterMuSeq = cms.Sequence(muSelSeq * (ZeeSelector + WenuSelector + EleSCSelector) * WZSelector)
