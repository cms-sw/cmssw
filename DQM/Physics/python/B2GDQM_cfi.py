import FWCore.ParameterSet.Config as cms

B2GDQM = cms.EDAnalyzer(
    "B2GDQM",

    #Trigger Results
    triggerResultsCollection = cms.InputTag("TriggerResults", "", "HLT"),

    PFJetCorService          = cms.string("ak4PFL1FastL2L3"),

    jetLabels = cms.VInputTag(
        'ak4PFJets',
        'ak4PFJetsCHS',
        'ca8PFJetsCHS',
        'ca8PFJetsCHSPruned',
        'cmsTopTagPFJetsCHS'
        ),
    jetPtMins = cms.vdouble(
        50.,
        50.,
        50.,
        50.,
        100.
        ),
    pfMETCollection          = cms.InputTag("pfMet")


)
