import FWCore.ParameterSet.Config as cms

def customizePixelTracksForTriplets(process):

  from HLTrigger.Configuration.common import producers_by_type, esproducers_by_type
  producers = ['CAHitNtupletAlpakaPhase1@alpaka','CAHitNtupletAlpakaPhase2@alpaka']
  for name in producers:
  	for producer in producers_by_type(process, name):
        	producer.minHitsPerNtuplet = 3
  
  for esproducer in esproducers_by_type(process,"CAGeometryESProducer@alpaka"):

    print(esproducer)

    esproducer.pairGraph = [  0, 1, 0, 4, 0,
        7, 1, 2, 1, 4,
        1, 7, 4, 5, 7,
        8, 2, 3, 2, 4,
        2, 7, 5, 6, 8,
        9, 0, 2, 1, 3,
        0, 5, 0, 8, 
        4, 6, 7, 9 ]
    esproducer.startingPairs = [i for i in range(8)] + [13, 14, 15, 16, 17, 18, 19]
    esproducer.phiCuts = [522, 730, 730, 522, 626,
        626, 522, 522, 626, 626,
        626, 522, 522, 522, 522,
        522, 522, 522, 522]
    esproducer.minZ = [-20., 0., -30., -22., 10., 
       -30., -70., -70., -22., 15., 
       -30, -70., -70., -20., -22., 
       0, -30., -70., -70.]
    esproducer.maxZ = [20., 30., 0., 22., 30., 
       -10., 70., 70., 22., 30., 
       -15., 70., 70., 20., 22., 
       30., 0., 70., 70.]
    esproducer.maxR = [20., 9., 9., 20., 7., 
       7., 5., 5., 20., 6., 
       6., 5., 5., 20., 20., 
       9., 9., 9., 9.]

  return process
