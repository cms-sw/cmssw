import FWCore.ParameterSet.Config as cms
import copy

from RecoTauTag.RecoTau.PFRecoTauDiscriminationByECALIsolation_cfi import pfRecoTauDiscriminationByECALIsolation 
from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadPion

pfRecoTauDiscriminationByECALIsolationUsingLeadingPion = copy.deepcopy(pfRecoTauDiscriminationByECALIsolation)

# Require a lead pion (charged OR neutral) instead of strictly a leading track
pfRecoTauDiscriminationByECALIsolationUsingLeadingPion.Prediscriminants = requireLeadPion
