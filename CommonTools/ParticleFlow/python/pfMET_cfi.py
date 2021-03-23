import FWCore.ParameterSet.Config as cms

# Clone pfMET producer from RecoMET
from RecoMET.METProducers.pfMet_cfi import *

# Should the name be changed se it is similar to pfMet from reco ??
pfMET = pfMet.clone(
    alias = "pfMET",

    # Use PF2PAT cleaned jet collection (pfJets) for MET significance
    # instead of standard collection (ak4PFJets)?
    # It requires that MET is produced at the end.
    srcJets = "pfJets"
)
# print 'PF2PAT: Jet collection used for pfMET significance: ', pfMET.jets
