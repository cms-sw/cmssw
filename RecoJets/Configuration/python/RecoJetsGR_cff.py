import FWCore.ParameterSet.Config as cms



# Standard set:
from RecoJets.Configuration.RecoJets_cff import *

#
# special R=0.15 IC jets:
ak15CaloJets = ak4CaloJets.clone( rParam = cms.double(0.15) )


recoJetsGR = cms.Sequence(ak15CaloJets+kt4CaloJets+kt6CaloJets+ak4CaloJets+ak4CaloJets+ak8CaloJets+sisCone5CaloJets+sisCone7CaloJets)

