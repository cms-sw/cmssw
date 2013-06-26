import FWCore.ParameterSet.Config as cms
import copy

# module to select generator level tau-decays
# See https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePhysicsCutParser
# on how to use the cut-string


# require generator level hadrons produced within muon acceptance
iterativeCone5GenJetsEta25 = cms.EDFilter("GenJetSelector",
     src = cms.InputTag("iterativeCone5GenJets"),
     cut = cms.string('abs(eta) < 2.5'),
     filter = cms.bool(False)
)

# require generator level hadrons produced in tau-decay to have transverse momentum above threshold
iterativeCone5GenJetsPt5Cumulative = cms.EDFilter("GenJetSelector",
     src = cms.InputTag("iterativeCone5GenJetsEta25"),
     cut = cms.string('pt > 5.'),
     filter = cms.bool(False)
)
