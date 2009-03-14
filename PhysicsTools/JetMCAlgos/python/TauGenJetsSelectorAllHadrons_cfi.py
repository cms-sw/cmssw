import FWCore.ParameterSet.Config as cms
import copy

# module to select generator level tau-decays
# See https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePhysicsCutParser
# on how to use the cut-string

#--------------------------------------------------------------------------------
# selection of tau --> hadrons nu_tau decays
#--------------------------------------------------------------------------------

# require generated tau to decay hadronically
selectedGenTauDecaysToHadrons = cms.EDFilter("TauGenJetDecayModeSelector",
     src = cms.InputTag("tauGenJets"),
     select = cms.vstring('oneProng0Pi0', 'oneProng1Pi0', 'oneProng2Pi0', 'oneProngOther',
                          'threeProng0Pi0', 'threeProng1Pi0', 'threeProngOther', 'rare'),
     filter = cms.bool(False)
)

# require generator level hadrons produced in tau-decay to be within muon acceptance
selectedGenTauDecaysToHadronsEta25Cumulative = cms.EDFilter("GenJetSelector",
     src = cms.InputTag("selectedGenTauDecaysToHadrons"),
     cut = cms.string('abs(eta) < 2.5'),
     filter = cms.bool(False)
)

# require generator level hadrons produced in tau-decay to have transverse momentum above threshold
selectedGenTauDecaysToHadronsPt5Cumulative = cms.EDFilter("GenJetSelector",
     src = cms.InputTag("selectedGenTauDecaysToHadronsEta25Cumulative"),
     cut = cms.string('pt > 5.'),
     filter = cms.bool(False)
)
