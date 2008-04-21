import FWCore.ParameterSet.Config as cms

# Full Event content
HLTMuonFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltL2MuonSeeds_*_*', 
        'keep *_hltL2Muons_*_*', 
        'keep *_hltL3Muons_*_*', 
        'keep *_hltL2MuonCandidates_*_*', 
        'keep *_hltL3MuonCandidates_*_*', 
        'keep *_hltL2MuonIsolations_*_*', 
        'keep *_hltL3MuonIsolations_*_*')
)
# RECO content
HLTMuonRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltL2MuonSeeds_*_*', 
        'keep *_hltL2Muons_*_*', 
        'keep *_hltL3Muons_*_*', 
        'keep *_hltL2MuonCandidates_*_*', 
        'keep *_hltL3MuonCandidates_*_*', 
        'keep *_hltL2MuonIsolations_*_*', 
        'keep *_hltL3MuonIsolations_*_*')
)
# AOD content
HLTMuonAOD = cms.PSet(
    triggerCollections = cms.VInputTag(cms.InputTag("hltL3MuonCandidates"), cms.InputTag("hltL2MuonCandidates")),
    triggerFilters = cms.VInputTag(cms.InputTag("hltSingleMuIsoL3IsoFiltered"), cms.InputTag("hltSingleMuNoIsoL3PreFiltered"), cms.InputTag("hltDiMuonIsoL3IsoFiltered"), cms.InputTag("hltDiMuonNoIsoL3PreFiltered"), cms.InputTag("hltZMML3Filtered"), 
        cms.InputTag("hltJpsiMML3Filtered"), cms.InputTag("hltUpsilonMML3Filtered"), cms.InputTag("hltMultiMuonNoIsoL3PreFiltered"), cms.InputTag("hltSameSignMuL3IsoFiltered"), cms.InputTag("hltExclDiMuonIsoL3IsoFiltered"), 
        cms.InputTag("hltSingleMuPrescale3L3PreFiltered"), cms.InputTag("hltSingleMuPrescale5L3PreFiltered"), cms.InputTag("hltSingleMuPrescale710L3PreFiltered"), cms.InputTag("hltSingleMuPrescale77L3PreFiltered"), cms.InputTag("hltDiMuonNoIsoL3PreFilteredRelaxedVtx2cm"), 
        cms.InputTag("hltDiMuonNoIsoL3PreFilteredRelaxedVtx2mm"), cms.InputTag("hltSingleMuNoIsoL3PreFilteredRelaxedVtx2cm"), cms.InputTag("hltSingleMuNoIsoL3PreFilteredRelaxedVtx2mm"), cms.InputTag("hltSingleMuStartupL2PreFiltered")),
    outputCommands = cms.untracked.vstring()
)

