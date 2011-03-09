import FWCore.ParameterSet.Config as cms

# Clone pfMET producer from RecoMET  
from RecoMET.METProducers.PFMET_cfi import *

# Should the name be changed se it is similar to pfMet from reco ??
pfMET = pfMet.clone()

# Use PF2PAT cleaned jet collection (pfJets) for MET significance
# instead of standard collection (ak5PFJets)?
# It requires that MET is produced at the end.
# Currently impossible as metsignificance requires that pfMET and pfJets
# are produced using the same input collection. Modifications needed in
# SignPFSpecificAlgo::addPFJets, SignPFSpecificAlgo::addPFCandidate
#pfMET.jets = cms.InputTag("pfJets")

print 'PF2PAT: Jet collection used for pfMET significance: ', pfMET.jets 
