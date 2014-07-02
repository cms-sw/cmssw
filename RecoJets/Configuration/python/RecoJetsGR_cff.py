import FWCore.ParameterSet.Config as cms



# Standard set:
from RecoJets.Configuration.RecoJets_cff import *

#
# special R=0.15 IC jets:
iterativeCone15CaloJets = iterativeCone5CaloJets.clone( rParam = cms.double(0.15) )


recoJetsGR = cms.Sequence(fixedGridRhoFastjetAllCalo+iterativeCone15CaloJets+kt4CaloJets+kt6CaloJets+iterativeCone5CaloJets+ak4CaloJets+ak7CaloJets+sisCone5CaloJets+sisCone7CaloJets)

