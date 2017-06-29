import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.slimming.slimmedPhotons_cfi import *

slimmedOOTPhotons = slimmedPhotons.clone()
slimmedOOTPhotons.src = cms.InputTag("selectedPatOOTPhotons")

slimmedOOTPhotons.linkToPackedPFCandidates = cms.bool(False)
slimmedOOTPhotons.recoToPFMap = cms.InputTag("")
slimmedOOTPhotons.packedPFCandidates = cms.InputTag("")
slimmedOOTPhotons.saveNonZSClusterShapes = cms.string("(r9()>0.8)") # save additional user floats: (sigmaIetaIeta,sigmaIphiIphi,sigmaIetaIphi,r9,e1x5_over_e5x5)_NoZS 

slimmedOOTPhotons.modifyPhotons = cms.bool(False)

