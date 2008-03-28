import FWCore.ParameterSet.Config as cms

import copy
from AnalysisAlgos.TrackInfoProducer.TrackInfoProducer_cfi import *
trackinfoRSTIFTIBTOB = copy.deepcopy(trackinfo)
import copy
from AnalysisAlgos.TrackInfoProducer.TrackInfoProducer_cfi import *
trackinfoCTFTIFTIBTOB = copy.deepcopy(trackinfo)
import copy
from AnalysisAlgos.TrackInfoProducer.TrackInfoProducer_cfi import *
trackinfoCosmicTFTIFTIBTOB = copy.deepcopy(trackinfo)
trackinfoTIFTIBTOB = cms.Sequence(trackinfoRSTIFTIBTOB*trackinfoCTFTIFTIBTOB*trackinfoCosmicTFTIFTIBTOB)
rsWithMaterialTracksTIFTIBTOB.TrajectoryInEvent = True
ctfWithMaterialTracksTIFTIBTOB.TrajectoryInEvent = True
cosmictrackfinderTIFTIBTOB.TrajInEvents = True
trackinfoRSTIFTIBTOB.cosmicTracks = 'rsWithMaterialTracksTIFTIBTOB'
trackinfoRSTIFTIBTOB.rechits = 'rsWithMaterialTracksTIFTIBTOB'
trackinfoCTFTIFTIBTOB.cosmicTracks = 'ctfWithMaterialTracksTIFTIBTOB'
trackinfoCTFTIFTIBTOB.rechits = 'ctfWithMaterialTracksTIFTIBTOB'
trackinfoCosmicTFTIFTIBTOB.cosmicTracks = 'cosmictrackfinderTIFTIBTOB'
trackinfoCosmicTFTIFTIBTOB.rechits = 'cosmictrackfinderTIFTIBTOB'

