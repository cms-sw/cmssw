import FWCore.ParameterSet.Config as cms

from RecoJets.Configuration.RecoGenJets_cff import *

"""
Build PFTauDecayModes containing Monte Carlo objects
They either contain the visible decay products of taus selected by 
the 'mcDecayedTaus' package or all of the consituents of a QCD GenJet
"""

mcDecayedTaus = cms.EDFilter("PdgIdAndStatusCandViewSelector",
    status = cms.vint32(2),
    src = cms.InputTag("genParticles"),
    pdgId = cms.vint32(15, -15)
)

makeMCTauDecayModes = cms.EDProducer("TruthTauDecayModeProducer",
    totalEtaCut = cms.double(2.5),
    inputTag = cms.InputTag("mcDecayedTaus"),
    leadTrackEtaCut = cms.double(2.5),
    leadTrackPtCut = cms.double(-1.0),
    totalPtCut = cms.double(5.0),
    iAmSignal = cms.bool(True)
)

makeMCQCDTauDecayModes = makeMCTauDecayModes.clone(
    totalEtaCut = cms.double(2.5),
    inputTag = cms.InputTag("ak4GenJets"),
    leadTrackEtaCut = cms.double(2.5),
    leadTrackPtCut = cms.double(-1.0),
    totalPtCut = cms.double(5.0),
    iAmSignal = cms.bool(False)
)

makeMC = cms.Sequence(mcDecayedTaus*makeMCTauDecayModes)

makeMCQCD = cms.Sequence(makeMCQCDTauDecayModes)

