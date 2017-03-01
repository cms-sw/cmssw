import FWCore.ParameterSet.Config as cms

from GeneratorInterface.GenFilters.CosmicGenFilterHelix_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import *

cosmicInPixelLoose = cosmicInTracker.clone()

cosmicInPixelLoose.radius = cms.double(20.0) ## i.e. twice 
cosmicInPixelLoose.minZ = cms.double(-100.0) ## the RECO SP
cosmicInPixelLoose.maxZ = cms.double(100.0)  ## skim cuts


