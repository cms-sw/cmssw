import FWCore.ParameterSet.Config as cms



# Standard set:
from RecoJets.Configuration.RecoJets_cff import *

recoJetsGRTask = cms.Task(fixedGridRhoFastjetAllCalo, ak4CaloJets)
recoJetsGR = cms.Sequence(recoJetsGRTask)

