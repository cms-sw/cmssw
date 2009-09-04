import FWCore.ParameterSet.Config as cms

import copy

from RecoTauTag.RecoTau.PFRecoTauDiscriminationByTrackIsolation_cfi import pfRecoTauDiscriminationByTrackIsolation 
from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadPion

pfRecoTauDiscriminationByTrackIsolationUsingLeadingPion = copy.deepcopy(pfRecoTauDiscriminationByTrackIsolation)

# Require a lead pion (charged OR neutral) instead of strictly a leading track
pfRecoTauDiscriminationByTrackIsolationUsingLeadingPion.Prediscriminants = requireLeadPion
