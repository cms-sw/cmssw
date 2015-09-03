import FWCore.ParameterSet.Config as cms

#Hotline skim parameters
singleMuonCut = 600
doubleMuonCut = 300
tripleMuonCut = 150
singleElectronCut = 900
doubleElectronCut = 300
tripleElectronCut = 100
singlePhotonCut = 1200
doublePhotonCut = 700
triplePhotonCut = 300
singleJetCut = 1600
doubleJetCut = 1400
multiJetCut = 100
multiJetNJets = 8
htCut = 4000
dimuonMassCut = 800
dielectronMassCut = 800
diEMuMassCut = 800

#MET hotline skim parameters
pfMetCut = 300
caloMetCut = 300
condPFMetCut = 100 #PF MET cut for large Calo/PF skim
condCaloMetCut = 100 #Calo MET cut for large PF/Calo skim
caloOverPFRatioCut = 2 #cut on Calo MET / PF MET
PFOverCaloRatioCut = 2 #cut on PF MET / Calo MET

## select events with at least one good PV
pvFilterHotLine = cms.EDFilter(
    "VertexSelector",
    src = cms.InputTag("offlinePrimaryVertices"),
    cut = cms.string("!isFake && ndof > 4 && abs(z) <= 24 && position.Rho <= 2"),
    filter = cms.bool(True),   # otherwise it won't filter the events, just produce an empty vertex collection.
)

## apply HBHE Noise filter
from CommonTools.RecoAlgos.HBHENoiseFilter_cfi import HBHENoiseFilter
from CommonTools.RecoAlgos.HBHENoiseFilterResultProducer_cfi import HBHENoiseFilterResultProducer

#CSC beam halo filter
from RecoMET.METFilters.CSCTightHaloFilter_cfi import *

#one muon
singleMuonSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("muons"),
    cut = cms.string( "isGlobalMuon() & pt() > "+str(singleMuonCut) )
)
singleMuonFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("singleMuonSelector"),
    minNumber = cms.uint32(1)
)
seqHotlineSkimSingleMuon = cms.Sequence(pvFilterHotLine * CSCTightHaloFilter * HBHENoiseFilterResultProducer * HBHENoiseFilter * singleMuonSelector * singleMuonFilter)

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
seqHotlineSkimDoubleMuon = cms.Sequence(pvFilterHotLine * CSCTightHaloFilter * HBHENoiseFilterResultProducer * HBHENoiseFilter * doubleMuonSelector * doubleMuonFilter)

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
seqHotlineSkimTripleMuon = cms.Sequence(pvFilterHotLine * CSCTightHaloFilter * HBHENoiseFilterResultProducer * HBHENoiseFilter * tripleMuonSelector * tripleMuonFilter)

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
seqHotlineSkimSingleElectron = cms.Sequence(pvFilterHotLine * CSCTightHaloFilter * HBHENoiseFilterResultProducer * HBHENoiseFilter * singleElectronSelector * singleElectronFilter)

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
seqHotlineSkimDoubleElectron = cms.Sequence(pvFilterHotLine * CSCTightHaloFilter * HBHENoiseFilterResultProducer * HBHENoiseFilter * doubleElectronSelector * doubleElectronFilter)

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
seqHotlineSkimTripleElectron = cms.Sequence(pvFilterHotLine * CSCTightHaloFilter * HBHENoiseFilterResultProducer * HBHENoiseFilter * tripleElectronSelector * tripleElectronFilter)

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
seqHotlineSkimSinglePhoton = cms.Sequence(pvFilterHotLine * CSCTightHaloFilter * HBHENoiseFilterResultProducer * HBHENoiseFilter * singlePhotonSelector * singlePhotonFilter)

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
seqHotlineSkimDoublePhoton = cms.Sequence(pvFilterHotLine * CSCTightHaloFilter * HBHENoiseFilterResultProducer * HBHENoiseFilter * doublePhotonSelector * doublePhotonFilter)

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
seqHotlineSkimTriplePhoton = cms.Sequence(pvFilterHotLine * CSCTightHaloFilter * HBHENoiseFilterResultProducer * HBHENoiseFilter * triplePhotonSelector * triplePhotonFilter)

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
seqHotlineSkimSingleJet = cms.Sequence(pvFilterHotLine * CSCTightHaloFilter * HBHENoiseFilterResultProducer * HBHENoiseFilter * singleJetSelector * singleJetFilter)

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
seqHotlineSkimDoubleJet = cms.Sequence(pvFilterHotLine * CSCTightHaloFilter * HBHENoiseFilterResultProducer * HBHENoiseFilter * doubleJetSelector * doubleJetFilter)

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
seqHotlineSkimMultiJet = cms.Sequence(pvFilterHotLine * CSCTightHaloFilter * HBHENoiseFilterResultProducer * HBHENoiseFilter * multiJetSelector * multiJetFilter)

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
seqHotlineSkimHT = cms.Sequence(pvFilterHotLine * CSCTightHaloFilter * HBHENoiseFilterResultProducer * HBHENoiseFilter * htMht * htSelector * htFilter)

#high-mass dileptons
dimuonsHotLine = cms.EDProducer(
    "CandViewShallowCloneCombiner",
    decay = cms.string("muons muons"),
    checkCharge = cms.bool(False),
    cut = cms.string("daughter(0).pt>150 & daughter(1).pt>150 & mass > "+str(dimuonMassCut)),
)
dielectrons = cms.EDProducer(
    "CandViewShallowCloneCombiner",
    decay = cms.string("gedGsfElectrons gedGsfElectrons"),
    checkCharge = cms.bool(False),
    cut = cms.string("daughter(0).pt>150 & daughter(1).pt>150 & mass > "+str(dielectronMassCut)),
)
diEMu = cms.EDProducer(
    "CandViewShallowCloneCombiner",
    decay = cms.string("muons gedGsfElectrons"),
    checkCharge = cms.bool(False),
    cut = cms.string("daughter(0).pt>150 & daughter(1).pt>150 & mass > "+str(diEMuMassCut)),
)
dimuonMassFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("dimuonsHotLine"),
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

seqHotlineSkimMassiveDimuon = cms.Sequence(pvFilterHotLine * CSCTightHaloFilter * HBHENoiseFilterResultProducer * HBHENoiseFilter * dimuonsHotLine * dimuonMassFilter)
seqHotlineSkimMassiveDielectron = cms.Sequence(pvFilterHotLine * CSCTightHaloFilter * HBHENoiseFilterResultProducer * HBHENoiseFilter * dielectrons * dielectronMassFilter)
seqHotlineSkimMassiveEMu = cms.Sequence(pvFilterHotLine * CSCTightHaloFilter * HBHENoiseFilterResultProducer * HBHENoiseFilter * diEMu * diEMuMassFilter)

## select events with high pfMET
pfMETSelectorHotLine = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("pfMet"),
    cut = cms.string( "pt()>"+str(pfMetCut) )
)

pfMETCounterHotLine = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("pfMETSelectorHotLine"),
    minNumber = cms.uint32(1),
)

seqHotlineSkimPFMET = cms.Sequence(
   pvFilterHotLine*
   CSCTightHaloFilter*
   HBHENoiseFilterResultProducer*
   HBHENoiseFilter*
   pfMETSelectorHotLine*
   pfMETCounterHotLine
)

## select events with high caloMET
caloMETSelectorHotLine = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("caloMetM"),
    cut = cms.string( "pt()>"+str(caloMetCut) )
)

caloMETCounterHotLine = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("caloMETSelectorHotLine"),
    minNumber = cms.uint32(1),
)

seqHotlineSkimCaloMET = cms.Sequence(
   pvFilterHotLine*
   CSCTightHaloFilter*
   HBHENoiseFilterResultProducer*
   HBHENoiseFilter*
   caloMETSelectorHotLine*
   caloMETCounterHotLine
)

## select events with extreme PFMET/CaloMET ratio
CondMETSelectorHotLine = cms.EDProducer(
   "CandViewShallowCloneCombiner",
   decay = cms.string("pfMet caloMetM"),
   cut = cms.string("(daughter(0).pt/daughter(1).pt > "+str(PFOverCaloRatioCut)+" && daughter(1).pt > "+str(condCaloMetCut)+") || (daughter(1).pt/daughter(0).pt > "+str(caloOverPFRatioCut)+" && daughter(0).pt > "+str(condPFMetCut)+" )  " )
)

CondMETCounterHotLine = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("CondMETSelectorHotLine"),
    minNumber = cms.uint32(1),
)

seqHotlineSkimCondMET = cms.Sequence(
   pvFilterHotLine*
   CSCTightHaloFilter*
   HBHENoiseFilterResultProducer*
   HBHENoiseFilter*
   CondMETSelectorHotLine*
   CondMETCounterHotLine
)
