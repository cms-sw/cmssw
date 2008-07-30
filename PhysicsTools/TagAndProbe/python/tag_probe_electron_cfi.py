import FWCore.ParameterSet.Config as cms

# 
#  GsfElectrons  ################
# 
from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *
# 
# Cut-based Robust electron ID  ######
# 
from EgammaAnalysis.ElectronIDProducers.electronId_cfi import *
import EgammaAnalysis.ElectronIDProducers.electronId_cfi
cutbasedRobustElectron = EgammaAnalysis.ElectronIDProducers.electronId_cfi.electronId.clone()
#  Calculate efficiency for *SuperClusters passing as GsfElectron* 
#
#  Tag           =  isolated GsfElectron with Robust ID, passing HLT, and  
#                    within the fiducial volume of ECAL
#  Probe --> Passing Probe   =  
#  SC --> GsfElectron --> isolation --> id --> Trigger
#
# 
#  SuperClusters  ################
# 
HybridSuperClusters = cms.EDProducer("EcalCandidateProducer",
    src = cms.InputTag("correctedHybridSuperClusters"),
    particleType = cms.string('gamma')
)

EndcapSuperClusters = cms.EDProducer("EcalCandidateProducer",
    src = cms.InputTag("correctedEndcapSuperClustersWithPreshower"),
    particleType = cms.string('gamma')
)

EBSuperClusters = cms.EDFilter("CandViewSelector",
    src = cms.InputTag("HybridSuperClusters"),
    cut = cms.string('abs( eta ) < 1.4442')
)

EESuperClusters = cms.EDFilter("CandViewSelector",
    src = cms.InputTag("EndcapSuperClusters"),
    cut = cms.string('abs( eta ) > 1.560 & abs( eta ) < 2.5')
)

allSuperClusters = cms.EDFilter("CandViewMerger",
    src = cms.VInputTag(cms.InputTag("EBSuperClusters"), cms.InputTag("EESuperClusters"))
)

# Duplicate Removal 
gsfElectrons = cms.EDFilter("ElectronDuplicateRemover",
    src = cms.untracked.string('pixelMatchGsfElectrons'),
    ptMin = cms.untracked.double(20.0),
    EndcapMinEta = cms.untracked.double(1.56),
    ptMax = cms.untracked.double(100.0),
    BarrelMaxEta = cms.untracked.double(1.4442),
    EndcapMaxEta = cms.untracked.double(2.5)
)

# 
#  isolation  ################
# 
isolatedElectronCands = cms.EDProducer("IsolatedElectronCandProducer",
    absolut = cms.bool(False),
    trackProducer = cms.InputTag("generalTracks"),
    isoCut = cms.double(0.2),
    intRadius = cms.double(0.02),
    electronProducer = cms.InputTag("gsfElectrons"),
    extRadius = cms.double(0.2),
    ptMin = cms.double(1.5),
    maxVtxDist = cms.double(0.1)
)

cutbasedRobustElectronCands = cms.EDProducer("eidCandProducer",
    ElectronIDAssociationProducer = cms.string('cutbasedRobustElectron'),
    InputProducer = cms.string('isolatedElectronCands')
)

# 
# Trigger  ##################
# 
HLTRobustElectronCands = cms.EDProducer("eTriggerCandProducer",
    triggerEventTag = cms.untracked.InputTag("hltTriggerSummaryAOD"),
    triggerDelRMatch = cms.untracked.double(0.3),
    hltTag = cms.untracked.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter"),
    InputProducer = cms.string('cutbasedRobustElectronCands')
)

# 
#  All Tag / Probe Collections  ###############
# 
# SuperCluster
theSuperClusters = cms.EDFilter("CandViewSelector",
    src = cms.InputTag("allSuperClusters"),
    cut = cms.string('et  > 20.0')
)

#  GsfElectron
gsfSelection = cms.EDFilter("PixelMatchGsfElectronSelector",
    src = cms.InputTag("gsfElectrons"),
    cut = cms.string('et > 0.0')
)

theGsfElectrons = cms.EDProducer("GsfElectronShallowCloneProducer",
    src = cms.InputTag("gsfSelection")
)

#  isolation
isoSelection = cms.EDFilter("PixelMatchGsfElectronSelector",
    src = cms.InputTag("isolatedElectronCands"),
    cut = cms.string('et > 0.0')
)

theIsolation = cms.EDProducer("GsfElectronShallowCloneProducer",
    src = cms.InputTag("isoSelection")
)

#  id 
idSelection = cms.EDFilter("PixelMatchGsfElectronSelector",
    src = cms.InputTag("cutbasedRobustElectronCands"),
    cut = cms.string('et > 0.0')
)

theId = cms.EDProducer("GsfElectronShallowCloneProducer",
    src = cms.InputTag("idSelection")
)

#  trigger
hltSelection = cms.EDFilter("PixelMatchGsfElectronSelector",
    src = cms.InputTag("HLTRobustElectronCands"),
    cut = cms.string('et > 0.0')
)

theHLT = cms.EDProducer("GsfElectronShallowCloneProducer",
    src = cms.InputTag("hltSelection")
)

# 
#  All Tag / Probe Association Maps  ###############
# 
# Remember that tag will always be "theHLT" collection.
#
# Probe can be one of the following collections: 
# "theSuperClusters", "theGsfElectrons", "theIsolation", "theId".
# 
# Passing Probe can be one of the following collections: 
# "theGsfElectrons", "theIsolation", "theId", "theHLT".
#
tpMapSuperClusters = cms.EDProducer("TagProbeProducer",
    MassMaxCut = cms.untracked.double(100.0),
    TagCollection = cms.InputTag("theHLT"),
    MassMinCut = cms.untracked.double(80.0),
    ProbeCollection = cms.InputTag("theSuperClusters")
)

tpMapGsfElectrons = cms.EDProducer("TagProbeProducer",
    MassMaxCut = cms.untracked.double(100.0),
    TagCollection = cms.InputTag("theHLT"),
    MassMinCut = cms.untracked.double(80.0),
    ProbeCollection = cms.InputTag("theGsfElectrons")
)

tpMapIsolation = cms.EDProducer("TagProbeProducer",
    MassMaxCut = cms.untracked.double(100.0),
    TagCollection = cms.InputTag("theHLT"),
    MassMinCut = cms.untracked.double(80.0),
    ProbeCollection = cms.InputTag("theIsolation")
)

tpMapId = cms.EDProducer("TagProbeProducer",
    MassMaxCut = cms.untracked.double(100.0),
    TagCollection = cms.InputTag("theHLT"),
    MassMinCut = cms.untracked.double(80.0),
    ProbeCollection = cms.InputTag("theId")
)

# 
#  All Truth-matched collections  ###################
# 
# find generator particles matching by DeltaR
SuperClustersMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(11),
    src = cms.InputTag("theSuperClusters"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("genParticles")
)

GsfElectronsMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(11),
    src = cms.InputTag("theGsfElectrons"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("genParticles")
)

IsolationMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(11),
    src = cms.InputTag("theIsolation"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("genParticles")
)

IdMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(11),
    src = cms.InputTag("theId"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("genParticles")
)

HLTMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(11),
    src = cms.InputTag("theHLT"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("genParticles")
)

lepton_cands = cms.Sequence(genParticles+HybridSuperClusters+EndcapSuperClusters+EBSuperClusters+EESuperClusters+allSuperClusters+gsfElectrons+isolatedElectronCands+electronId+cutbasedRobustElectron+cutbasedRobustElectronCands+HLTRobustElectronCands+theSuperClusters+gsfSelection+theGsfElectrons+isoSelection+theIsolation+idSelection+theId+hltSelection+theHLT+tpMapSuperClusters+tpMapGsfElectrons+tpMapIsolation+tpMapId+SuperClustersMatch+GsfElectronsMatch+IsolationMatch+IdMatch+HLTMatch)
cutbasedRobustElectron.electronProducer = 'isolatedElectronCands'
cutbasedRobustElectron.doPtdrId = False
cutbasedRobustElectron.doCutBased = True
cutbasedRobustElectron.algo_psets.append(cms.PSet(
    electronQuality = cms.string('robust')
))

