import FWCore.ParameterSet.Config as cms

# When re-running HLT, this function will provide a set of output commands to
# provide the minimal content necessary for input to HLTMuonValidator.
def outputCommands(hlt_name):
    return cms.untracked.vstring(
        'drop *_*_*_*',
        'keep *_genParticles_*_*',
        'keep recoMuons_muons_*_RECO',
        'keep *_hltL1extraParticles_*_%s' % hlt_name,
        'keep *_hltL2Muons_*_%s' % hlt_name,
        'keep *_hltL3Muons_*_%s' % hlt_name,
        'keep *_hltL2MuonCandidates_*_%s' % hlt_name,
        'keep *_hltL3MuonCandidates_*_%s' % hlt_name,
        'keep *_hltTriggerSummaryRAW_*_%s' % hlt_name,
        'keep *_TriggerResults_*_%s' % hlt_name,
        'keep *_hltL2MuonSeeds_*_%s' % hlt_name,
        'keep *_hltL3TrajectorySeed_*_%s' % hlt_name,
        )
