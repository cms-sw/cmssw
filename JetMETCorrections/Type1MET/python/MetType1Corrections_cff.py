import FWCore.ParameterSet.Config as cms

# File: MetCorrections.cff
# Author: R. Cavanaugh
# Date: 08.08.2006
#
# Met corrections for the icone5, mcone5, and mcone7 MC corrected jets.
corMetType1Icone5 = cms.EDFilter("Type1MET",
    inputUncorJetsLabel = cms.string('iterativeCone5CaloJets'),
    jetEMfracLimit = cms.double(0.9),
    metType = cms.string('CaloMET'),
    jetPTthreshold = cms.double(20.0),
    inputUncorMetLabel = cms.string('met'),
    corrector = cms.string('MCJetCorrectorIcone5')
)

corMetType1Mcone5 = cms.EDFilter("Type1MET",
    inputUncorJetsLabel = cms.string('midPointCone5CaloJets'),
    jetEMfracLimit = cms.double(0.9),
    metType = cms.string('CaloMET'),
    jetPTthreshold = cms.double(20.0),
    inputUncorMetLabel = cms.string('met'),
    corrector = cms.string('MCJetCorrectorMcone5')
)

corMetType1Mcone7 = cms.EDFilter("Type1MET",
    inputUncorJetsLabel = cms.string('midPointCone7CaloJets'),
    jetEMfracLimit = cms.double(0.9),
    metType = cms.string('CaloMET'),
    jetPTthreshold = cms.double(20.0),
    inputUncorMetLabel = cms.string('met'),
    corrector = cms.string('MCJetCorrectorMcone7')
)

MetType1Corrections = cms.Sequence(corMetType1Icone5*corMetType1Mcone5*corMetType1Mcone7)

