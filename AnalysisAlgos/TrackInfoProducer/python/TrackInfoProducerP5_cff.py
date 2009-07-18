import FWCore.ParameterSet.Config as cms

import copy
from AnalysisAlgos.TrackInfoProducer.TrackInfoProducer_cfi import *
trackinfoRSP5 = copy.deepcopy(trackinfo)
import copy
from AnalysisAlgos.TrackInfoProducer.TrackInfoProducer_cfi import *
trackinfoCTFP5 = copy.deepcopy(trackinfo)
import copy
from AnalysisAlgos.TrackInfoProducer.TrackInfoProducer_cfi import *
trackinfoCosmicTFP5 = copy.deepcopy(trackinfo)
trackinfoP5 = cms.Sequence(trackinfoRSP5*trackinfoCTFP5*trackinfoCosmicTFP5)
rsWithMaterialTracksP5.TrajectoryInEvent = True
ctfWithMaterialTracksP5.TrajectoryInEvent = True
cosmictrackfinderP5.TrajInEvents = True
trackinfoRSP5.cosmicTracks = 'rsWithMaterialTracksP5'
trackinfoRSP5.rechits = 'rsWithMaterialTracksP5'
trackinfoCTFP5.cosmicTracks = 'ctfWithMaterialTracksP5'
trackinfoCTFP5.rechits = 'ctfWithMaterialTracksP5'
trackinfoCosmicTFP5.cosmicTracks = 'cosmictrackfinderP5'
trackinfoCosmicTFP5.rechits = 'cosmictrackfinderP5'

