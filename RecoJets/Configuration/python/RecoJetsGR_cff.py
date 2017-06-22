import FWCore.ParameterSet.Config as cms



# Standard set:
from RecoJets.Configuration.RecoJets_cff import *

recoJetsGR = cms.Sequence(fixedGridRhoFastjetAllCalo+ak4CaloJets)

