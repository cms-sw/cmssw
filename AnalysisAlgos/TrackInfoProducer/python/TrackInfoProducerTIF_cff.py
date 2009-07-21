import FWCore.ParameterSet.Config as cms

import copy
from AnalysisAlgos.TrackInfoProducer.TrackInfoProducer_cfi import *
trackinfoRSTIF = copy.deepcopy(trackinfo)
import copy
from AnalysisAlgos.TrackInfoProducer.TrackInfoProducer_cfi import *
trackinfoCTFTIF = copy.deepcopy(trackinfo)
import copy
from AnalysisAlgos.TrackInfoProducer.TrackInfoProducer_cfi import *
trackinfoCosmicTFTIF = copy.deepcopy(trackinfo)
trackinfoTIF = cms.Sequence(trackinfoRSTIF*trackinfoCTFTIF*trackinfoCosmicTFTIF)
rsWithMaterialTracksTIF.TrajectoryInEvent = True
ctfWithMaterialTracksTIF.TrajectoryInEvent = True
cosmictrackfinderTIF.TrajInEvents = True
trackinfoRSTIF.cosmicTracks = 'rsWithMaterialTracksTIF'
trackinfoRSTIF.rechits = 'rsWithMaterialTracksTIF'
trackinfoCTFTIF.cosmicTracks = 'ctfWithMaterialTracksTIF'
trackinfoCTFTIF.rechits = 'ctfWithMaterialTracksTIF'
trackinfoCosmicTFTIF.cosmicTracks = 'cosmictrackfinderTIF'
trackinfoCosmicTFTIF.rechits = 'cosmictrackfinderTIF'

