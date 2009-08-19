import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
from PhysicsTools.HepMCCandAlgos.genParticleCandidates_cfi import *
# Muons
SelectGenMuons = cms.EDFilter("EtaPtMinPdgIdCandSelector",
    src = cms.InputTag("genParticleCandidates"),
    etaMin = cms.double(-5.0),
    etaMax = cms.double(5.0),
    ptMin = cms.double(10.0),
    pdgId = cms.vint32(13)
)

# Electrons
SelectGenElec = cms.EDFilter("EtaPtMinPdgIdCandSelector",
    src = cms.InputTag("genParticleCandidates"),
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
SelectGenForJets = cms.EDFilter("EtaPtMinCandSelector",
    src = cms.InputTag("CloneGenJets"),
    etaMin = cms.double(-5.0),
    etaMax = cms.double(-3.0),
    ptMin = cms.double(20.0)
)

# Tau
# Select generator tau particles
SelectGenTauJets = cms.EDFilter("EtaPtMinPdgIdCandSelector",
    src = cms.InputTag("genParticleCandidates"),
    etaMin = cms.double(-3.0),
    etaMax = cms.double(3.0),
    ptMin = cms.double(20.0),
    pdgId = cms.vint32(15)
)

# Missing Et
# Clone generator Met
CloneGenMet = cms.EDProducer("GenMETShallowCloneProducer",
    src = cms.InputTag("genMet")
)

# Select generator Met
SelectGenMet = cms.EDFilter("PtMinCandSelector",
    src = cms.InputTag("CloneGenMet"),
    ptMin = cms.double(10.0)
)

GenMuonSelection = cms.Sequence(genParticleCandidates*SelectGenMuons)
GenElecSelection = cms.Sequence(genParticleCandidates*SelectGenElec)
GenCenJetSelection = cms.Sequence(CloneGenJets*SelectGenCenJets)
GenForJetSelection = cms.Sequence(CloneGenJets*SelectGenForJets)
GenTauJetSelection = cms.Sequence(genParticleCandidates*SelectGenTauJets)
GenMetSelection = cms.Sequence(CloneGenMet*SelectGenMet)

