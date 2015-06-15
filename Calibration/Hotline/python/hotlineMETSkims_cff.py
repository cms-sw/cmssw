import FWCore.ParameterSet.Config as cms

pfMetCut = 300
caloMetCut = 300
condPFMetCut = 100 #PF MET cut for large Calo/PF skim
condCaloMetCut = 100 #Calo MET cut for large PF/Calo skim
caloOverPFRatioCut = 2 #cut on Calo MET / PF MET
PFOverCaloRatioCut = 2 #cut on PF MET / Calo MET

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
