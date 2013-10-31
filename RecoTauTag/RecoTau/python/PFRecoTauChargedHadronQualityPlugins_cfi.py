import FWCore.ParameterSet.Config as cms

'''

Configuration for ChargedHadron producer plugins.

Author: Chriastian Veelken, LLR


'''

isChargedPFCandidate = cms.PSet(
    name = cms.string('ChargedPFCandidate'),
    plugin = cms.string('PFRecoTauChargedHadronStringQuality'),
    selection = cms.string("algoIs('kChargedPFCandidate')"),
    selectionPassFunction = cms.string("-pt"), # CV: give preference to highest Pt candidate
    selectionFailValue = cms.double(1.e+3)
)

isTrack = isChargedPFCandidate.clone(
    selection = cms.string("algoIs('kTrack')")
)

isPFNeutralHadron = isChargedPFCandidate.clone(
    selection = cms.string("algoIs('kPFNeutralHadron')")
)
