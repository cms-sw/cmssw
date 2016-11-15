import FWCore.ParameterSet.Config as cms

from copy import deepcopy

#Geometry
# include used for track reconstruction 
# note that tracking is redone since we need updated hits and they 
# are not stored in the event!
# include "RecoTracker/TrackProducer/data/CTFFinalFitWithMaterial.cff"
from RecoParticleFlow.PFProducer.particleFlowBlock_cfi import *

