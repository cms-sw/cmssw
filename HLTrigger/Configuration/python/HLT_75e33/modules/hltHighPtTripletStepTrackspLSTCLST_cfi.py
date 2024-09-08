import FWCore.ParameterSet.Config as cms

from ..modules.hltHighPtTripletStepTracks_cfi import hltHighPtTripletStepTracks as _hltHighPtTripletStepTracks
hltHighPtTripletStepTrackspLSTCLST = _hltHighPtTripletStepTracks.clone( src = "hltHighPtTripletStepTrackCandidatespLSTCLST" )

