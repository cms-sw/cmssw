import FWCore.ParameterSet.Config as cms

def customizePixelTracksForTriplets(process):

  from HLTrigger.Configuration.common import producers_by_type
  for producer in producers_by_type(process, 'CAHitNtupletCUDA'):
        producer.includeJumpingForwardDoublets = True
        producer.minHitsPerNtuplet = 3
 
  return process
