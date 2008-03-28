import FWCore.ParameterSet.Config as cms

# Full Event content
HLTMuonFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*')
)
# RECO content
HLTMuonRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*')
)
# AOD content
HLTMuonAOD = cms.PSet(
    triggerCollections = cms.VInputTag(cms.InputTag("hltL3MuonCandidates"), cms.InputTag("hltL2MuonCandidates")),
    triggerFilters = cms.VInputTag(cms.InputTag("SingleMuIsoL3IsoFiltered"), cms.InputTag("SingleMuNoIsoL3PreFiltered"), cms.InputTag("DiMuonIsoL3IsoFiltered"), cms.InputTag("DiMuonNoIsoL3PreFiltered"), cms.InputTag("ZMML3Filtered"), cms.InputTag("JpsiMML3Filtered"), cms.InputTag("UpsilonMML3Filtered"), cms.InputTag("multiMuonNoIsoL3PreFiltered"), cms.InputTag("SameSignMuL3IsoFiltered"), cms.InputTag("ExclDiMuonIsoL3IsoFiltered"), cms.InputTag("SingleMuPrescale3L3PreFiltered"), cms.InputTag("SingleMuPrescale5L3PreFiltered"), cms.InputTag("SingleMuPrescale710L3PreFiltered"), cms.InputTag("SingleMuPrescale77L3PreFiltered"), cms.InputTag("DiMuonNoIsoL3PreFilteredRelaxedVtx2cm"), cms.InputTag("DiMuonNoIsoL3PreFilteredRelaxedVtx2mm"), cms.InputTag("SingleMuNoIsoL3PreFilteredRelaxedVtx2cm"), cms.InputTag("SingleMuNoIsoL3PreFilteredRelaxedVtx2mm"), cms.InputTag("SingleMuStartupL2PreFiltered")),
    outputCommands = cms.untracked.vstring()
)

