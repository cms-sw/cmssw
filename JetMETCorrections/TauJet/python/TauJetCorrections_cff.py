import FWCore.ParameterSet.Config as cms

# File: JetCorrections.cff
# Author: O. Kodolova
# Modified for CMSSW120 by: F. Blekman
# Date: 2/12/07
#
#
# TauJet corrections for the icone4 jets. (ORCA derived)
# 
TauJetCorrectorIcone4 = cms.ESSource("TauJetCorrectionService",
    TauTriggerType = cms.int32(1),
    tagName = cms.string('IterativeCone0.4_standardCMSSW_120_tau'),
    label = cms.string('TauJetCorrectorIcone4')
)

TauJetCorJetIcone4 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("iterativeCone4CaloJets"),
    correctors = cms.vstring('TauJetCorrectorIcone4'),
    alias = cms.untracked.string('TauJetCorJetIcone4')
)

TauJetCorrections = cms.Sequence(TauJetCorJetIcone4)

