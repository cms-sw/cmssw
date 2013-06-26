import FWCore.ParameterSet.Config as cms

#from RecoBTag.MCTools.mcJetFlavour_cff import *
from  PhysicsTools.JetMCAlgos.CaloJetsMCFlavour_cfi import *

btagCalibPath = cms.Path(myPartons*iterativeCone5Flavour)


