import FWCore.ParameterSet.Config as cms

# This modifier merges the initialStep and highPtTripletStep iterations
# to a single iteration using Patatrack pixel tracks with >3 hits as seeds
singleIterPatatrack = cms.Modifier()
