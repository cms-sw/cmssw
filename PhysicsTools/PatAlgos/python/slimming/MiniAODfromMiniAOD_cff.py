import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.slimming.modifyPrimaryPhysicsObjects_cff import *
from PhysicsTools.PatAlgos.slimming.MicroEventContent_cff import *

EIsequence = cms.Sequence( modifyPrimaryPhysicsObjects )
