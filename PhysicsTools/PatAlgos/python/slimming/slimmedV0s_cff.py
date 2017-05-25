import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.slimming.slimmedSecondaryVertices_cfi      import *
slimmedKshortVertices=slimmedSecondaryVertices.clone(src=cms.InputTag("generalV0Candidates","Kshort"))
slimmedLambdaVertices=slimmedSecondaryVertices.clone(src=cms.InputTag("generalV0Candidates","Lambda"))
