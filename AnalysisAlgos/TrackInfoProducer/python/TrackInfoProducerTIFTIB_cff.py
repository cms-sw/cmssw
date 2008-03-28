import FWCore.ParameterSet.Config as cms

import copy
from AnalysisAlgos.TrackInfoProducer.TrackInfoProducer_cfi import *
trackinfoRSTIFTIB = copy.deepcopy(trackinfo)
import copy
from AnalysisAlgos.TrackInfoProducer.TrackInfoProducer_cfi import *
trackinfoCTFTIFTIB = copy.deepcopy(trackinfo)
import copy
from AnalysisAlgos.TrackInfoProducer.TrackInfoProducer_cfi import *
trackinfoCosmicTFTIFTIB = copy.deepcopy(trackinfo)
trackinfoTIFTIB = cms.Sequence(trackinfoRSTIFTIB*trackinfoCTFTIFTIB*trackinfoCosmicTFTIFTIB)
rsWithMaterialTracksTIFTIB.TrajectoryInEvent = True
ctfWithMaterialTracksTIFTIB.TrajectoryInEvent = True
cosmictrackfinderTIFTIB.TrajInEvents = True
trackinfoRSTIFTIB.cosmicTracks = 'rsWithMaterialTracksTIFTIB'
trackinfoRSTIFTIB.rechits = 'rsWithMaterialTracksTIFTIB'
trackinfoCTFTIFTIB.cosmicTracks = 'ctfWithMaterialTracksTIFTIB'
trackinfoCTFTIFTIB.rechits = 'ctfWithMaterialTracksTIFTIB'
trackinfoCosmicTFTIFTIB.cosmicTracks = 'cosmictrackfinderTIFTIB'
trackinfoCosmicTFTIFTIB.rechits = 'cosmictrackfinderTIFTIB'

