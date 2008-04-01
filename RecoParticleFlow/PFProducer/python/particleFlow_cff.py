import FWCore.ParameterSet.Config as cms

#Geometry
from Geometry.CaloEventSetup.CaloGeometry_cfi import *
# include used for track reconstruction 
# note that tracking is redone since we need updated hits and they 
# are not stored in the event!
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff import *
from RecoParticleFlow.PFProducer.particleFlow_cfi import *


