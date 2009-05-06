import FWCore.ParameterSet.Config as cms

#Geometry
# include "Geometry/CaloEventSetup/data/CaloGeometry.cfi"
# include used for track reconstruction 
# note that tracking is redone since we need updated hits and they 
# are not stored in the event!
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff import *
from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
from RecoParticleFlow.PFProducer.particleFlowSimParticle_cfi import *

