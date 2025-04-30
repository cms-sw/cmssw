import FWCore.ParameterSet.Config as cms

##################################################################
# AlCaReco for track based calibration using V0s
##################################################################
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
ALCARECOTkAlV0sHLTFilter = hltHighLevel.clone()
ALCARECOTkAlV0sHLTFilter.andOr = True ## choose logical OR between Triggerbits
ALCARECOTkAlV0sHLTFilter.throw = False ## dont throw on unknown path names
ALCARECOTkAlV0sHLTFilter.HLTPaths = ['HLT_*']
#ALCARECOTkAlV0sHLTFilter.eventSetupPathsKey = 'TkAlV0s'

##################################################################
# Select events with at least one V0
##################################################################
from DQM.TrackingMonitorSource.v0EventSelector_cfi import *
ALCARECOTkAlV0sKShortEventSelector = v0EventSelector.clone(
    vertexCompositeCandidates = "generalV0Candidates:Kshort"  
)
ALCARECOTkAlV0sLambdaEventSelector = v0EventSelector.clone(
    vertexCompositeCandidates = "generalV0Candidates:Lambda"  
)

##################################################################
# Tracks from the selected vertex
#################################################################
import Alignment.CommonAlignmentProducer.AlignmentTracksFromVertexCompositeCandidateSelector_cfi as TracksFromV0
ALCARECOTkAlKShortTracks = TracksFromV0.AlignmentTracksFromVertexCompositeCandidateSelector.clone(
    vertexCompositeCandidates = 'generalV0Candidates:Kshort'
)

ALCARECOTkAlLambdaTracks = TracksFromV0.AlignmentTracksFromVertexCompositeCandidateSelector.clone(
    vertexCompositeCandidates = 'generalV0Candidates:Lambda'
)

##################################################################
# Sequence
##################################################################
seqALCARECOTkAlK0s = cms.Sequence(ALCARECOTkAlV0sHLTFilter + ALCARECOTkAlV0sKShortEventSelector + ALCARECOTkAlKShortTracks)
seqALCARECOTkAlLambdas = cms.Sequence(ALCARECOTkAlV0sHLTFilter + ALCARECOTkAlV0sLambdaEventSelector + ALCARECOTkAlLambdaTracks)
-- dummy change --
