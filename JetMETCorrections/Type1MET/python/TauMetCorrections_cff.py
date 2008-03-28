import FWCore.ParameterSet.Config as cms

# File: TauMetCorrections.cff
# Authors: C. N. Nguyen , A. Gurrola
# Date: 10.22.2007
#
# Met corrections for PFTaus
PFJetsCaloJetsDeltaMet = cms.EDFilter("TauMET",
    InputPFJetsLabel = cms.string('iterativeCone5PFJets'),
    JetMatchDeltaR = cms.double(0.2),
    InputCaloJetsLabel = cms.string('iterativeCone5CaloJets'),
    UseCorrectedJets = cms.bool(False),
    correctorLabel = cms.string('MCJetCorrectorIcone5')
)

PFJetsCorrCaloJetsDeltaMet = cms.EDFilter("TauMET",
    InputPFJetsLabel = cms.string('iterativeCone5PFJets'),
    JetMatchDeltaR = cms.double(0.2),
    InputCaloJetsLabel = cms.string('iterativeCone5CaloJets'),
    UseCorrectedJets = cms.bool(True),
    correctorLabel = cms.string('MCJetCorrectorIcone5')
)

MetTauCorrections = cms.Sequence(PFJetsCaloJetsDeltaMet*PFJetsCorrCaloJetsDeltaMet)

