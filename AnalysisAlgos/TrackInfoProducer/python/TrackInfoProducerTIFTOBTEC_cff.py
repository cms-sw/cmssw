import FWCore.ParameterSet.Config as cms

import copy
from AnalysisAlgos.TrackInfoProducer.TrackInfoProducer_cfi import *
trackinfoRSTIFTOBTEC = copy.deepcopy(trackinfo)
import copy
from AnalysisAlgos.TrackInfoProducer.TrackInfoProducer_cfi import *
trackinfoCTFTIFTOBTEC = copy.deepcopy(trackinfo)
import copy
from AnalysisAlgos.TrackInfoProducer.TrackInfoProducer_cfi import *
trackinfoCosmicTFTIFTOBTEC = copy.deepcopy(trackinfo)
trackinfoTIFTOBTEC = cms.Sequence(trackinfoRSTIFTOBTEC*trackinfoCTFTIFTOBTEC*trackinfoCosmicTFTIFTOBTEC)
rsWithMaterialTracksTIFTOBTEC.TrajectoryInEvent = True
ctfWithMaterialTracksTIFTOBTEC.TrajectoryInEvent = True
cosmictrackfinderTIFTOBTEC.TrajInEvents = True
trackinfoRSTIFTOBTEC.cosmicTracks = 'rsWithMaterialTracksTIFTOBTEC'
trackinfoRSTIFTOBTEC.rechits = 'rsWithMaterialTracksTIFTOBTEC'
trackinfoCTFTIFTOBTEC.cosmicTracks = 'ctfWithMaterialTracksTIFTOBTEC'
trackinfoCTFTIFTOBTEC.rechits = 'ctfWithMaterialTracksTIFTOBTEC'
trackinfoCosmicTFTIFTOBTEC.cosmicTracks = 'cosmictrackfinderTIFTOBTEC'
trackinfoCosmicTFTIFTOBTEC.rechits = 'cosmictrackfinderTIFTOBTEC'

