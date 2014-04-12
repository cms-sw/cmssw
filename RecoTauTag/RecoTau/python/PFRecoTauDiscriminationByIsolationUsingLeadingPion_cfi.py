import FWCore.ParameterSet.Config as cms

import copy

from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolation_cfi import pfRecoTauDiscriminationByIsolation 
from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadPion

pfRecoTauDiscriminationByIsolationUsingLeadingPion = copy.deepcopy(pfRecoTauDiscriminationByIsolation)

# Require a lead pion (charged OR neutral) instead of strictly a leading track
pfRecoTauDiscriminationByIsolationUsingLeadingPion.Prediscriminants = requireLeadPion
