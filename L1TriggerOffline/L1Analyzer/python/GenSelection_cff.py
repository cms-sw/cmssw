import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *

# Muons
SelectGenMuons = cms.EDFilter("EtaPtMinPdgIdCandViewSelector",
    src = cms.InputTag("genParticles"),
    etaMin = cms.double(-5.0),
    etaMax = cms.double(5.0),
    ptMin = cms.double(10.0),
    pdgId = cms.vint32(13)
)

# Electrons
SelectGenElec = cms.EDFilter("EtaPtMinPdgIdCandViewSelector",
    src = cms.InputTag("genParticles"),
    etaMin = cms.double(-3.0),
    etaMax = cms.double(3.0),
    ptMin = cms.double(10.0),
    pdgId = cms.vint32(11)
)

# Jets
# Clone generator jets
CloneGenJets = cms.EDProducer("GenJetShallowCloneProducer",
    src = cms.InputTag("iterativeCone5GenJets")
)

# Select central generator jets
SelectGenCenJets = cms.EDFilter("EtaPtMinCandSelector",
    src = cms.InputTag("CloneGenJets"),
    etaMin = cms.double(-3.0),
    etaMax = cms.double(3.0),
    ptMin = cms.double(20.0)
)

# Select forward generator jets
SelectGenForJets = cms.EDFilter("CandViewSelector",
    src = cms.InputTag("CloneGenJets"),
    cut = cms.string("(-5.0 < eta < -3.0 | 3.0 < eta < 5.0) & pt > 20.0")                            
)

# Tau
# Select generator tau particles
SelectGenTauJets = cms.EDFilter("EtaPtMinPdgIdCandViewSelector",
    src = cms.InputTag("genParticles"),
    etaMin = cms.double(-3.0),
    etaMax = cms.double(3.0),
    ptMin = cms.double(20.0),
    pdgId = cms.vint32(15)
)

# Missing Et
# Clone generator Met
CloneGenMet = cms.EDProducer("GenMETShallowCloneProducer",
    src = cms.InputTag("genMetTrue")
)

# Select generator Met
SelectGenMet = cms.EDFilter("PtMinCandSelector",
    src = cms.InputTag("CloneGenMet"),
    ptMin = cms.double(10.0)
)

# Select generator jets
SelectGenJets = cms.EDFilter("PtMinCandSelector",
    src = cms.InputTag("CloneGenJets"),
    ptMin = cms.double(20.0)
)

# Merge generator jets with generator tau jets to make a collection of all jets
MergeGenJets = cms.EDProducer("CandViewMerger",
    src = cms.VInputTag("SelectGenJets","SelectGenTauJets")
)

GenMuonSelection = cms.Sequence(genParticles*SelectGenMuons)
GenElecSelection = cms.Sequence(genParticles*SelectGenElec)
GenCenJetSelection = cms.Sequence(CloneGenJets*SelectGenCenJets)
GenForJetSelection = cms.Sequence(CloneGenJets*SelectGenForJets)
GenTauJetSelection = cms.Sequence(genParticles*SelectGenTauJets)
GenMetSelection = cms.Sequence(CloneGenMet*SelectGenMet)

GenJetSelection = cms.Sequence(CloneGenJets*SelectGenJets)
