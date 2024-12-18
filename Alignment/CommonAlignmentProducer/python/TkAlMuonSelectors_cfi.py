import FWCore.ParameterSet.Config as cms

# original selection:
# 'isGlobalMuon & isTrackerMuon & numberOfMatches > 1 & globalTrack.hitPattern.numberOfValidMuonHits > 0
#  & abs(eta) < 2.5 & globalTrack.normalizedChi2 < 20.'

from Alignment.CommonAlignmentProducer.alignmentGoodIdMuonSelector_cfi import alignmentGoodIdMuonSelector
TkAlGoodIdMuonSelector = alignmentGoodIdMuonSelector.clone(src = 'muons',
                                                           requireGlobal = True,
                                                           requireTracker = True,
                                                           minMatches = 1,
                                                           minMuonHits = 0,
                                                           maxEta = 2.5,
                                                           maxChi2 = 20,
                                                           filter = True)
# original selection:
# '(isolationR03().sumPt + isolationR03().emEt + isolationR03().hadEt)/pt  < 0.15'

from Alignment.CommonAlignmentProducer.alignmentRelCombIsoMuonSelector_cfi import alignmentRelCombIsoMuonSelector
TkAlRelCombIsoMuonSelector = alignmentRelCombIsoMuonSelector.clone(src = cms.InputTag('TkAlGoodIdMuonSelector'),
                                                                   filter = True,
                                                                   relCombIsoCut = 0.15,
                                                                   useTrackerOnlyIsolation = False)

# Define a common sequence to be imported in ALCARECOs
seqALCARECOTkAlRelCombIsoMuons = cms.Sequence(TkAlGoodIdMuonSelector+TkAlRelCombIsoMuonSelector)

## FIXME: these are needed for ALCARECO production in CMSSW_14_0_X
## to avoid loosing in efficiency. To be reviewed after muon reco is fixed

# original selection:
# '(isGlobalMuon & isTrackerMuon & numberOfMatches > 1 & globalTrack.hitPattern.numberOfValidMuonHits > 0
#  & abs(eta) < 2.5 & globalTrack.normalizedChi2 < 20.) || (abs(eta) > 2.3 & abs(eta) < 3.0 & numberOfMatches >= 0 & isTrackerMuon)'

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(TkAlGoodIdMuonSelector,
                       useSecondarySelection = True, # to recover tracks passing through GE0
                       secondaryEtaLow = 2.3,
                       secondaryEtaHigh = 3,
                       secondaryMinMatches = 0,
                       secondaryRequireTracker = True)

# original selection:
# '(isolationR03().sumPt)/pt < 0.1'

phase2_common.toModify(TkAlRelCombIsoMuonSelector,
                       relCombIsoCut = 0.10,
                       useTrackerOnlyIsolation = True)  # only tracker isolation
