import FWCore.ParameterSet.Config as cms

def customizePixelTracksForTriplets(process):

  from HLTrigger.Configuration.common import producers_by_type
  producers = ['CAHitNtupletCUDA','CAHitNtupletCUDAPhase1','CAHitNtupletCUDAPhase2','CAHitNtupletAlpakaPhase1@alpaka','CAHitNtupletAlpakaPhase2@alpaka']
  for name in producers:
  	for producer in producers_by_type(process, name):
        	producer.includeJumpingForwardDoublets = True
        	producer.minHitsPerNtuplet = 3
 
  return process
