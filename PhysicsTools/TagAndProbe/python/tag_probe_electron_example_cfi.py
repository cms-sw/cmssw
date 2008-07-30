import FWCore.ParameterSet.Config as cms

#
#  Electron tag and probe example. Modify this script to define 
# your own tag and probe collections 
#
#  Tag           =  isolated GsfElectron Loose ID with fiducial cuts
#  Probe         =  GsfElectron with fiducial cuts
#  Passing Probe =  GsfElectron with fiducial cuts
#
from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *
from EgammaAnalysis.ElectronIDProducers.electronId_cfi import *
import EgammaAnalysis.ElectronIDProducers.electronId_cfi
# CutBased: Robust electron ID
cutbasedRobustElectron = EgammaAnalysis.ElectronIDProducers.electronId_cfi.electronId.clone()
# Duplicate Removal 
uniqueElectrons = cms.EDFilter("ElectronDuplicateRemover",
    src = cms.string('pixelMatchGsfElectrons')
)

# isolation 
isolatedElectronCands = cms.EDProducer("IsolatedElectronCandProducer",
    absolut = cms.bool(False),
    trackProducer = cms.InputTag("generalTracks"),
    isoCut = cms.double(0.2),
    intRadius = cms.double(0.02),
    electronProducer = cms.InputTag("uniqueElectrons"),
    extRadius = cms.double(0.2),
    ptMin = cms.double(1.5),
    maxVtxDist = cms.double(0.1)
)

# electron ID Candidate collection ############
cutbasedRobustElectronCands = cms.EDProducer("eidCandProducer",
    ElectronIDAssociationProducer = cms.string('cutbasedRobustElectron'),
    InputProducer = cms.string('isolatedElectronCands')
)

# HLT ################
HLTRobustElectronCands = cms.EDProducer("eTriggerCandProducer",
    triggerEventTag = cms.untracked.InputTag("hltTriggerSummaryAOD"),
    triggerDelRMatch = cms.untracked.double(0.3),
    hltTag = cms.untracked.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter"),
    InputProducer = cms.string('cutbasedRobustElectronCands')
)

#
# Make the input candidate collections
#tag candidates = GsfElectron with fiducial cuts
tagElectrons = cms.EDFilter("GsfElectronSelector",
    src = cms.InputTag("HLTRobustElectronCands"),
    cut = cms.string('pt > 20.0 & ( abs( eta ) < 1.4442 | (abs( eta ) > 1.560 & abs( eta ) < 2.5))')
)

tagCands = cms.EDProducer("GsfElectronShallowCloneProducer",
    src = cms.InputTag("tagElectrons")
)

#allProbe candidates = SuperClusters with fiducial cuts
allProbeEBSuperClusters = cms.EDProducer("ConcreteEcalCandidateProducer",
    src = cms.InputTag("correctedHybridSuperClusters"),
    particleType = cms.string('gamma')
)

allProbeEESuperClusters = cms.EDProducer("ConcreteEcalCandidateProducer",
    src = cms.InputTag("correctedEndcapSuperClustersWithPreshower"),
    particleType = cms.string('gamma')
)

allProbeSuperClusters = cms.EDFilter("CandViewMerger",
    src = cms.VInputTag(cms.InputTag("allProbeEBSuperClusters"), cms.InputTag("allProbeEESuperClusters"))
)

allProbeCands = cms.EDFilter("CandViewSelector",
    src = cms.InputTag("allProbeSuperClusters"),
    cut = cms.string('pt > 20.0 & ( abs( eta ) < 1.4442 | (abs( eta ) > 1.560 & abs( eta ) < 2.5))')
)

#passProbe candidates = isolated GsfElectron with fiducial cuts
passProbeElectrons = cms.EDFilter("GsfElectronSelector",
    src = cms.InputTag("isolatedElectronCands"),
    cut = cms.string('pt > 20.0 & ( abs( eta ) < 1.4442 | (abs( eta ) > 1.560 & abs( eta ) < 2.5))')
)

passProbeCands = cms.EDProducer("GsfElectronShallowCloneProducer",
    src = cms.InputTag("passProbeElectrons")
)

# Make the tag probe association map
tagProbeMap = cms.EDProducer("TagProbeProducer",
    MassMaxCut = cms.untracked.double(120.0),
    TagCollection = cms.InputTag("tagCands"),
    MassMinCut = cms.untracked.double(60.0),
    ProbeCollection = cms.InputTag("allProbeCands")
)

# find generator particles matching by DeltaR
tagMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(11),
    src = cms.InputTag("tagCands"),
    distMin = cms.double(0.25),
    matched = cms.InputTag("genParticles")
)

allProbeMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(11),
    src = cms.InputTag("allProbeCands"),
    distMin = cms.double(0.25),
    matched = cms.InputTag("genParticles")
)

passProbeMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(11),
    src = cms.InputTag("passProbeCands"),
    distMin = cms.double(0.25),
    matched = cms.InputTag("genParticles")
)

lepton_cands = cms.Sequence(genParticles+uniqueElectrons+isolatedElectronCands+electronId+cutbasedRobustElectron+cutbasedRobustElectronCands+HLTRobustElectronCands+tagElectrons+tagCands+allProbeEBSuperClusters+allProbeEESuperClusters+allProbeSuperClusters+allProbeCands+passProbeElectrons+passProbeCands+tagProbeMap+tagMatch+allProbeMatch+passProbeMatch)
cutbasedRobustElectron.electronProducer = 'isolatedElectronCands'
cutbasedRobustElectron.doPtdrId = False
cutbasedRobustElectron.doCutBased = True
cutbasedRobustElectron.algo_psets.append(cms.PSet(
    electronQuality = cms.string('robust')
))

