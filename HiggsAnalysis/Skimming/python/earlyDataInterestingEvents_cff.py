from FWCore.ParameterSet import Config as cms

## Common pre-filter sequence
oneGoodVertexFilter = cms.EDFilter("VertexSelector",
   src = cms.InputTag("offlinePrimaryVertices"),
   cut = cms.string("!isFake && ndof >= 4 && abs(z) <= 15 && position.Rho <= 2"),
   filter = cms.bool(True),   # otherwise it won't filter the events, just produce an empty vertex collection.
)
noScraping = cms.EDFilter("FilterOutScraping",
    applyfilter = cms.untracked.bool(True),
    debugOn = cms.untracked.bool(False), ## Or 'True' to get some per-event info
    numtrack = cms.untracked.uint32(10),
    thresh = cms.untracked.double(0.25)
)
earlyDataPreFilter = cms.Sequence(oneGoodVertexFilter * noScraping)

### Define basic "good" electrons and muons
## ---------------------------------
ELECTRON_BASE_CUT=("(fbrem > 0 &&" +
                   " eSuperClusterOverP < 3 &&" +
                   " hcalOverEcal < 0.15 &&" +
                   " abs(deltaPhiSuperClusterTrackAtVtx) < 0.10 &&" +
                   " abs(deltaEtaSuperClusterTrackAtVtx) < 0.02 &&" +
                   " (( isEB && sigmaIetaIeta > 0.008) ||" +  ### NOTE for the lazy people that don't read things carefully (e.g. myself)
                   "  (!isEB && sigmaIetaIeta > 0.02)) )");   ###  the cut is "sigma > something", and it's meant to cut away spikes.
                                                              ### it's not the usual electron id cut "sigma < something" to reject qcd.
goodElectrons = cms.EDFilter("GsfElectronRefSelector",
    src = cms.InputTag("gsfElectrons"),
    cut = cms.string(ELECTRON_BASE_CUT),
)

TM_ARBITRATION = "numberOfMatches('SegmentAndTrackArbitration')>0";
MUON_BASE_CUT="(isGlobalMuon || (isTrackerMuon && "+TM_ARBITRATION+"))"
goodMuons = cms.EDFilter("MuonRefSelector",
    src = cms.InputTag("muons"),
    cut = cms.string(MUON_BASE_CUT),
)

## High energy single objects
## ---------------------------------
highEnergyMuons = cms.EDFilter("MuonRefSelector",
    src = cms.InputTag("muons"),
    cut = cms.string(MUON_BASE_CUT +" && (pt > 20)"),
)
highEnergyElectrons = cms.EDFilter("GsfElectronRefSelector",
    src = cms.InputTag("gsfElectrons"),
    cut = cms.string(ELECTRON_BASE_CUT + "&& (pt > 20)"),
)

## Very loosely isolated leptons (e.g. for Ws)
## ---------------------------------
isolatedGoodMuons = cms.EDFilter("MuonRefSelector",
    src = cms.InputTag("muons"),
    cut = cms.string(MUON_BASE_CUT + " && ( isolationR03.hadEt + isolationR03.emEt < 10 ) && (isolationR03.emVetoEt + isolationR03.hadVetoEt < 15) && ( isolationR03.sumPt < 10 )"),
)
isolatedGoodElectrons = cms.EDFilter("GsfElectronRefSelector",
    src = cms.InputTag("gsfElectrons"),
    cut = cms.string(ELECTRON_BASE_CUT + "&& ( dr03EcalRecHitSumEt + dr03HcalTowerSumEt + dr03TkSumPt < 20 )"),
)

singleLeptons = cms.Sequence(
    goodMuons     +
    goodElectrons +
    highEnergyMuons     +
    highEnergyElectrons +
    isolatedGoodMuons      +
    isolatedGoodElectrons 
)


## Di-objects
## ---------------------------------
DILEPTON_MASS = "(mass > 10) && "
DILEPTON_PT   = " daughter(0).pt > 3 && daughter(1).pt > 3 && max( daughter(0).pt,  daughter(1).pt ) >= 5"
## For muons, require at least one of the two to be global. we don't check the charge in the inclusive selection
ONE_GLOBAL_MU = " && (daughter(0).isGlobalMuon || daughter(1).isGlobalMuon)"
diMuons = cms.EDProducer("CandViewShallowCloneCombiner",
    decay       = cms.string("goodMuons goodMuons"), 
    checkCharge = cms.bool(False),           # can be changed to goodElectrons@+ goodElectrons@- to keep only OS
    cut         = cms.string(DILEPTON_MASS + DILEPTON_PT + ONE_GLOBAL_MU),
)
## As a reference, we make also a small skim arond the J/Psi, to see if it works
diMuonsJPsi = cms.EDProducer("CandViewShallowCloneCombiner",
    decay           = cms.string("goodMuons@+ goodMuons@-"), 
    cut             = cms.string("(2.6 < mass < 3.6)  && daughter(0).isGlobalMuon && daughter(1).isGlobalMuon"),
)
## and one at very high mass, where the Z should dominate
diMuonsZ = cms.EDProducer("CandViewShallowCloneCombiner",
    decay           = cms.string("goodMuons@+ goodMuons@-"), 
    cut             = cms.string("(mass > 60) && " + DILEPTON_PT + ONE_GLOBAL_MU),
)

## For electrons, we don't enforce the opposite sign
diElectrons = cms.EDProducer("CandViewShallowCloneCombiner",
    decay       = cms.string("goodElectrons goodElectrons"), # no sign specified: allow also the SS electrons in.
    checkCharge = cms.bool(False),           # can be changed to goodElectrons@+ goodElectrons@- to keep only OS
    cut         = cms.string(DILEPTON_MASS + DILEPTON_PT),
)
## As a reference, we make also a small skim arond the J/Psi, to see if it works
diElectronsJPsi = cms.EDProducer("CandViewShallowCloneCombiner",
    decay       = cms.string("goodElectrons goodElectrons"), # no sign specified: allow also the SS electrons in.
    checkCharge = cms.bool(False),           # can be changed to goodElectrons@+ goodElectrons@- to keep only OS
    cut         = cms.string("(2.6 < mass < 3.6) && daughter(0).pt > 3 && daughter(1).pt > 3"),
)
## and one at very high mass, where the Z should dominate
diElectronsZ = cms.EDProducer("CandViewShallowCloneCombiner",
    decay       = cms.string("goodElectrons goodElectrons"), # no sign specified: allow also the SS electrons in.
    checkCharge = cms.bool(False),           # can be changed to goodElectrons@+ goodElectrons@- to keep only OS
    cut         = cms.string("(mass > 60) && " + DILEPTON_PT + ONE_GLOBAL_MU),
)

## Then we make a E+Mu skim
crossLeptons  = cms.EDProducer("CandViewShallowCloneCombiner",
    decay       = cms.string("goodMuons goodElectrons"), # no sign specified: allow also the SS pairs in.
    checkCharge = cms.bool(False),           
    cut         = cms.string(DILEPTON_MASS + DILEPTON_PT),    
)

diLeptons = cms.Sequence(
    diMuons     + diMuonsJPsi     + diMuonsZ     +
    diElectrons + diElectronsJPsi + diElectronsZ +
    crossLeptons
)

## Isolated lepton plus MET, aka W candidate
## ---------------------------------
W_MU_PT  = 10; W_EL_PT  = 15; W_PF_MET = 15; W_TC_MET = 15; MT_CUT = 45
## Note: the 'mt()' method doesn't compute the transverse mass correctly, so we have to do it by hand.
MT="sqrt(2*daughter(0).pt*daughter(1).pt*(1 - cos(daughter(0).phi - daughter(1).phi)))"
recoWMNfromPf = cms.EDProducer("CandViewShallowCloneCombiner",
    decay       = cms.string("isolatedGoodMuons@+ pfMet"), 
    cut         = cms.string(("daughter(0).pt > %f && daughter(1).pt > %f && "+MT+" > %f") % (W_MU_PT, W_PF_MET, MT_CUT)),
)
recoWMNfromTc = cms.EDProducer("CandViewShallowCloneCombiner",
    decay       = cms.string("isolatedGoodMuons@+ tcMet"), 
    cut         = cms.string(("daughter(0).pt > %f && daughter(1).pt > %f && "+MT+" > %f") % (W_MU_PT, W_TC_MET, MT_CUT)),
)
recoWENfromPf = cms.EDProducer("CandViewShallowCloneCombiner",
    decay       = cms.string("isolatedGoodElectrons@+ pfMet"), 
    cut         = cms.string(("daughter(0).pt > %f && daughter(1).pt > %f && "+MT+" > %f") % (W_EL_PT, W_PF_MET, MT_CUT)),
)
recoWENfromTc = cms.EDProducer("CandViewShallowCloneCombiner",
    decay       = cms.string("isolatedGoodElectrons@+ tcMet"), 
    cut         = cms.string(("daughter(0).pt > %f && daughter(1).pt > %f && "+MT+" > %f") % (W_EL_PT, W_TC_MET, MT_CUT)),
)
recoWs = cms.Sequence(
    recoWMNfromPf + recoWMNfromTc +
    recoWENfromPf + recoWENfromTc 
)

## Tri and 4-objects
## ---------------------------------
triLeptons =  cms.EDProducer("CandViewShallowCloneCombiner",
    decay       = cms.string("L L L"), 
    checkCharge = cms.bool(False),           
    cut         = cms.string("mass > 10 && min(min(daughter(0).pt,daughter(1).pt),daughter(2).pt) > 2 && max(max(daughter(0).pt,daughter(1).pt),daughter(2).pt) >= 5"),    
)
triLeptonsMuMuMu  = triLeptons.clone(decay = "goodMuons goodMuons goodMuons")
triLeptonsMuMuEl  = triLeptons.clone(decay = "goodMuons goodMuons goodElectrons")
triLeptonsMuElEl  = triLeptons.clone(decay = "goodMuons goodElectrons goodElectrons")
triLeptonsElElEl  = triLeptons.clone(decay = "goodElectrons goodElectrons goodElectrons")

quadLeptons4Mu    = crossLeptons.clone(decay = "diMuons diMuons")
quadLeptons2Mu2El = crossLeptons.clone(decay = "diMuons diElectrons")
quadLeptons4El    = crossLeptons.clone(decay = "diElectrons diElectrons")

multiLeptons = cms.Sequence(
    triLeptonsMuMuMu + triLeptonsMuMuEl  + triLeptonsMuElEl + triLeptonsElElEl +
    quadLeptons4Mu   + quadLeptons2Mu2El + quadLeptons4El
)

earlyDataInterestingEvents = cms.Sequence(
    earlyDataPreFilter +
    singleLeptons      +
    diLeptons          +
    recoWs             +
    multiLeptons       
)


