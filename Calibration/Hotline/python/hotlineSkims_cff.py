import FWCore.ParameterSet.Config as cms

#Hotline skim parameters
singleMuonCut = 500
doubleMuonCut = 300
tripleMuonCut = 100
singleElectronCut = 500
doubleElectronCut = 300
tripleElectronCut = 100
singlePhotonCut = 800
doublePhotonCut = 400
triplePhotonCut = 200
singleJetCut = 1500
doubleJetCut = 500
multiJetCut = 100
multiJetNJets = 8
htCut = 4000
dimuonMassCut = 500
dielectronMassCut = 500
diEMuMassCut = 500

#MET hotline skim parameters
pfMetCut = 300
caloMetCut = 300
condPFMetCut = 100 #PF MET cut for large Calo/PF skim
condCaloMetCut = 100 #Calo MET cut for large PF/Calo skim
caloOverPFRatioCut = 2 #cut on Calo MET / PF MET
PFOverCaloRatioCut = 2 #cut on PF MET / Calo MET

#one muon
singleMuonSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("muons"),
    cut = cms.string( "pt() > "+str(singleMuonCut) )
)
singleMuonFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("singleMuonSelector"),
    minNumber = cms.uint32(1)
)
seqHotlineSkimSingleMuon = cms.Sequence(singleMuonSelector * singleMuonFilter)

#two muons
doubleMuonSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("muons"),
    cut = cms.string( "pt() > "+str(doubleMuonCut) )
)
doubleMuonFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("doubleMuonSelector"),
    minNumber = cms.uint32(2)
)
seqHotlineSkimDoubleMuon = cms.Sequence(doubleMuonSelector * doubleMuonFilter)

#three muons
tripleMuonSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("muons"),
    cut = cms.string( "pt() > "+str(tripleMuonCut) )
)
tripleMuonFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("tripleMuonSelector"),
    minNumber = cms.uint32(3)
)
seqHotlineSkimTripleMuon = cms.Sequence(tripleMuonSelector * tripleMuonFilter)

#one electron
singleElectronSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("gedGsfElectrons"),
    cut = cms.string( "pt() > "+str(singleElectronCut) )
)
singleElectronFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("singleElectronSelector"),
    minNumber = cms.uint32(1)
)
seqHotlineSkimSingleElectron = cms.Sequence(singleElectronSelector * singleElectronFilter)

#two electrons
doubleElectronSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("gedGsfElectrons"),
    cut = cms.string( "pt() > "+str(doubleElectronCut) )
)
doubleElectronFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("doubleElectronSelector"),
    minNumber = cms.uint32(2)
)
seqHotlineSkimDoubleElectron = cms.Sequence(doubleElectronSelector * doubleElectronFilter)

#three electrons
tripleElectronSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("gedGsfElectrons"),
    cut = cms.string( "pt() > "+str(tripleElectronCut) )
)
tripleElectronFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("tripleElectronSelector"),
    minNumber = cms.uint32(3)
)
seqHotlineSkimTripleElectron = cms.Sequence(tripleElectronSelector * tripleElectronFilter)

#one photon
singlePhotonSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("gedPhotons"),
    cut = cms.string( "pt() > "+str(singlePhotonCut) )
)
singlePhotonFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("singlePhotonSelector"),
    minNumber = cms.uint32(1)
)
seqHotlineSkimSinglePhoton = cms.Sequence(singlePhotonSelector * singlePhotonFilter)

#two photons
doublePhotonSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("gedPhotons"),
    cut = cms.string( "pt() > "+str(doublePhotonCut) )
)
doublePhotonFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("doublePhotonSelector"),
    minNumber = cms.uint32(2)
)
seqHotlineSkimDoublePhoton = cms.Sequence(doublePhotonSelector * doublePhotonFilter)

#three photons
triplePhotonSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("gedPhotons"),
    cut = cms.string( "pt() > "+str(triplePhotonCut) )
)
triplePhotonFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("triplePhotonSelector"),
    minNumber = cms.uint32(3)
)
seqHotlineSkimTriplePhoton = cms.Sequence(triplePhotonSelector * triplePhotonFilter)

#one jet
singleJetSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("ak4PFJets"),
    cut = cms.string( "pt() > "+str(singleJetCut) )
)
singleJetFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("singleJetSelector"),
    minNumber = cms.uint32(1)
)
seqHotlineSkimSingleJet = cms.Sequence(singleJetSelector * singleJetFilter)

#two jets
doubleJetSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("ak4PFJets"),
    cut = cms.string( "pt() > "+str(doubleJetCut) )
)
doubleJetFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("doubleJetSelector"),
    minNumber = cms.uint32(2)
)
seqHotlineSkimDoubleJet = cms.Sequence(doubleJetSelector * doubleJetFilter)

#many jets
multiJetSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("ak4PFJets"),
    cut = cms.string( "pt() > "+str(multiJetCut) )
)
multiJetFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("multiJetSelector"),
    minNumber = cms.uint32(multiJetNJets)
)
seqHotlineSkimMultiJet = cms.Sequence(multiJetSelector * multiJetFilter)

#HT
htMht = cms.EDProducer( "HLTHtMhtProducer",
    usePt = cms.bool( False ),
    minPtJetHt = cms.double( 40.0 ),
    maxEtaJetMht = cms.double( 5.0 ),
    minNJetMht = cms.int32( 0 ),
    jetsLabel = cms.InputTag( "ak4CaloJets" ),
    maxEtaJetHt = cms.double( 3.0 ),
    minPtJetMht = cms.double( 30.0 ),
    minNJetHt = cms.int32( 0 ),
    pfCandidatesLabel = cms.InputTag( "" ),
    excludePFMuons = cms.bool( False )
)
htSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("htMht"),
    cut = cms.string( "sumEt() > "+str(htCut) )
)
htFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("htSelector"),
    minNumber = cms.uint32(1)
)
seqHotlineSkimHT = cms.Sequence(htMht * htSelector * htFilter)

#high-mass dileptons
dimuons = cms.EDProducer(
    "CandViewShallowCloneCombiner",
    decay = cms.string("muons muons"),
    checkCharge = cms.bool(False),
    cut = cms.string("mass > "+str(dimuonMassCut)),
)
dielectrons = cms.EDProducer(
    "CandViewShallowCloneCombiner",
    decay = cms.string("gedGsfElectrons gedGsfElectrons"),
    checkCharge = cms.bool(False),
    cut = cms.string("mass > "+str(dielectronMassCut)),
)
diEMu = cms.EDProducer(
    "CandViewShallowCloneCombiner",
    decay = cms.string("muons gedGsfElectrons"),
    checkCharge = cms.bool(False),
    cut = cms.string("mass > "+str(diEMuMassCut)),
)
dimuonMassFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("dimuons"),
    minNumber = cms.uint32(1)
)
dielectronMassFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("dielectrons"),
    minNumber = cms.uint32(1)
)
diEMuMassFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("diEMu"),
    minNumber = cms.uint32(1)
)

seqHotlineSkimMassiveDimuon = cms.Sequence(dimuons * dimuonMassFilter)
seqHotlineSkimMassiveDielectron = cms.Sequence(dielectrons * dielectronMassFilter)
seqHotlineSkimMassiveEMu = cms.Sequence(diEMu * diEMuMassFilter)

## select events with at least one good PV
pvFilter = cms.EDFilter(
    "VertexSelector",
    src = cms.InputTag("offlinePrimaryVertices"),
    cut = cms.string("!isFake && ndof > 4 && abs(z) <= 24 && position.Rho <= 2"),
    filter = cms.bool(True),   # otherwise it won't filter the events, just produce an empty vertex collection.
)

## apply HBHE Noise filter
from CommonTools.RecoAlgos.HBHENoiseFilter_cfi import HBHENoiseFilter
from CommonTools.RecoAlgos.HBHENoiseFilterResultProducer_cfi import HBHENoiseFilterResultProducer

## select events with high pfMET
pfMETSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("pfMet"),
    cut = cms.string( "pt()>"+str(pfMetCut) )
)

pfMETCounter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("pfMETSelector"),
    minNumber = cms.uint32(1),
)

seqHotlineSkimPFMET = cms.Sequence(
   pvFilter*
   HBHENoiseFilterResultProducer*
   HBHENoiseFilter*
   pfMETSelector*
   pfMETCounter
)

## select events with high caloMET
caloMETSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("caloMetM"),
    cut = cms.string( "pt()>"+str(caloMetCut) )
)

caloMETCounter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("caloMETSelector"),
    minNumber = cms.uint32(1),
)

seqHotlineSkimCaloMET = cms.Sequence(
   pvFilter*
   HBHENoiseFilterResultProducer*
   HBHENoiseFilter*
   caloMETSelector*
   caloMETCounter
)

## select events with extreme PFMET/CaloMET ratio
CondMETSelector = cms.EDProducer(
   "CandViewShallowCloneCombiner",
   decay = cms.string("pfMet caloMetM"),
   cut = cms.string("(daughter(0).pt/daughter(1).pt > "+str(PFOverCaloRatioCut)+" && daughter(1).pt > "+str(condCaloMetCut)+") || (daughter(1).pt/daughter(0).pt > "+str(caloOverPFRatioCut)+" && daughter(0).pt > "+str(condPFMetCut)+" )  " )
)

CondMETCounter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("CondMETSelector"),
    minNumber = cms.uint32(1),
)

seqHotlineSkimCondMET = cms.Sequence(
   pvFilter*
   HBHENoiseFilterResultProducer*
   HBHENoiseFilter*
   CondMETSelector*
   CondMETCounter
)
