import FWCore.ParameterSet.Config as cms

import copy
from AnalysisAlgos.TrackInfoProducer.TrackInfoProducer_cfi import *
trackinfoRS = copy.deepcopy(trackinfo)
import copy
from AnalysisAlgos.TrackInfoProducer.TrackInfoProducer_cfi import *
trackinfoCTF = copy.deepcopy(trackinfo)
import copy
from AnalysisAlgos.TrackInfoProducer.TrackInfoProducer_cfi import *
trackinfoCosmicTF = copy.deepcopy(trackinfo)
rsWithMaterialTracks.TrajectoryInEvent = True
ctfWithMaterialTracks.TrajectoryInEvent = True
cosmictrackfinder.TrajInEvents = True
trackinfoRS.cosmicTracks = 'rsWithMaterialTracks'
trackinfoRS.rechits = 'rsWithMaterialTracks'
trackinfoCTF.cosmicTracks = 'ctfWithMaterialTracks'
trackinfoCTF.rechits = 'ctfWithMaterialTracks'
trackinfoCosmicTF.cosmicTracks = 'cosmictrackfinder'
trackinfoCosmicTF.rechits = 'cosmictrackfinder'

