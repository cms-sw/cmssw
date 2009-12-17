import FWCore.ParameterSet.Config as cms

egHLTEleTrigRelEffQTests = cms.PSet (
    name=cms.string('Ele Rel Trig Eff'),
    qTestsToCheck = cms.vstring('_trigEffTo*gsfEle_trigCuts*_vs_et_*',
                                '_trigEffTo*gsfEle_trigCuts*_vs_eta_*',
                                '_trigEffTo*gsfEle_trigCuts*_vs_phi_*')
    )

egHLTPhoTrigRelEffQTests = cms.PSet (
    name=cms.string('Pho Rel Trig Eff'),
    qTestsToCheck = cms.vstring('_trigEffTo*pho_trigCuts*_vs_et_*',
                                '_trigEffTo*pho_trigCuts*_vs_eta_*',
                                '_trigEffTo*pho_trigCuts*_vs_phi_*')
    )
egHLTEleTrigTPEffQTests = cms.PSet (
    name=cms.string('Ele T&P Trig Eff'),
    qTestsToCheck = cms.vstring('_trigTagProbeEff_gsfEle_trigCuts*')
    )
egHLTTrigEleQTests = cms.PSet (
    name=cms.string('Triggered Eles'),
    qTestsToCheck = cms.vstring('*gsfEle_effVsEt_n1*',
                                '*gsfEle_effVsEta_n1*',
                                '*gsfEle_effVsEta_n1*')
    )
egHLTTrigPhoQTests = cms.PSet (
    name=cms.string('Triggered Phos'),
    qTestsToCheck = cms.vstring('*pho_effVsEt_n1*',
                                '*pho_effVsEta_n1*',
                                '*pho_effVsEta_n1*')
    )
