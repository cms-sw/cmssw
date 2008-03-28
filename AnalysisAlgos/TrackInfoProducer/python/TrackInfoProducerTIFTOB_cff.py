import FWCore.ParameterSet.Config as cms

import copy
from AnalysisAlgos.TrackInfoProducer.TrackInfoProducer_cfi import *
trackinfoRSTIFTOB = copy.deepcopy(trackinfo)
import copy
from AnalysisAlgos.TrackInfoProducer.TrackInfoProducer_cfi import *
trackinfoCTFTIFTOB = copy.deepcopy(trackinfo)
import copy
from AnalysisAlgos.TrackInfoProducer.TrackInfoProducer_cfi import *
trackinfoCosmicTFTIFTOB = copy.deepcopy(trackinfo)
trackinfoTIFTOB = cms.Sequence(trackinfoRSTIFTOB*trackinfoCTFTIFTOB*trackinfoCosmicTFTIFTOB)
rsWithMaterialTracksTIFTOB.TrajectoryInEvent = True
ctfWithMaterialTracksTIFTOB.TrajectoryInEvent = True
cosmictrackfinderTIFTOB.TrajInEvents = True
trackinfoRSTIFTOB.cosmicTracks = 'rsWithMaterialTracksTIFTOB'
trackinfoRSTIFTOB.rechits = 'rsWithMaterialTracksTIFTOB'
trackinfoCTFTIFTOB.cosmicTracks = 'ctfWithMaterialTracksTIFTOB'
trackinfoCTFTIFTOB.rechits = 'ctfWithMaterialTracksTIFTOB'
trackinfoCosmicTFTIFTOB.cosmicTracks = 'cosmictrackfinderTIFTOB'
trackinfoCosmicTFTIFTOB.rechits = 'cosmictrackfinderTIFTOB'

