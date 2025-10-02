import FWCore.ParameterSet.Config as cms

def customizePixelTracksForTriplets(process):
   
   from HLTrigger.Configuration.common import producers_by_type, esproducers_by_type
   names = ['CAHitNtupletAlpakaPhase1@alpaka','CAHitNtupletAlpakaPhase2@alpaka']
   
   for name in names:
      producers = producers_by_type(process, name)
      for producer in producers:
         producer.minHitsPerNtuplet = 3

         if name == 'CAHitNtupletAlpakaPhase1@alpaka':
          
            producer.avgHitsPerTrack    = 4.5      
            producer.avgCellsPerHit     = 27
            producer.avgCellsPerCell    = 0.071 
            producer.avgTracksPerCell   = 0.127 
            producer.maxNumberOfDoublets = str(512*1024) # this is actually low, should be ~630k, keeping the same for a fair comparison with master
            producer.maxNumberOfTuples   = str(32 * 1024) # this is on spot (µ+5*σ = 31.8k)

            producer.geometry.pairGraph = [  0, 1, 0, 4, 0,
               7, 1, 2, 1, 4,
               1, 7, 4, 5, 7,
               8, 2, 3, 2, 4,
               2, 7, 5, 6, 8,
               9, 0, 2, 1, 3,
               0, 5, 0, 8, 
               4, 6, 7, 9 ]

            nPairs = int(len(producer.geometry.pairGraph) / 2)
            producer.geometry.startingPairs = [i for i in range(8)] + [13, 14, 15, 16, 17, 18]

            producer.geometry.phiCuts = [522, 730, 730, 522, 626,
               626, 522, 522, 626, 626,
               626, 522, 522, 522, 522,
               522, 522, 522, 522]
            producer.geometry.minInner = [-20., 0., -30., -22., 10., 
               -30., -70., -70., -22., 15., 
               -30, -70., -70., -20., -22., 
               0, -30., -70., -70.]
            producer.geometry.maxInner = [20., 30., 0., 22., 30., 
               -10., 70., 70., 22., 30., 
               -15., 70., 70., 20., 22., 
               30., 0., 70., 70.]
            producer.geometry.maxDR = [20., 9., 9., 20., 7., 
               7., 5., 5., 20., 6., 
               6., 5., 5., 20., 20., 
               9., 9., 9., 9.]
            producer.geometry.minDZ = [-10000] * nPairs
            producer.geometry.maxDZ = [10000] * nPairs
            producer.geometry.minOuter = [-10000] * nPairs
            producer.geometry.maxOuter = [10000] * nPairs
            producer.geometry.ptCuts = [0.5] * nPairs
                   
         elif name == 'CAHitNtupletAlpakaPhase2@alpaka':

            producer.avgHitsPerTrack    = 6.5      
            producer.avgCellsPerHit     = 6 # actually this is the same, quads has the same graph at the moment
            producer.avgCellsPerCell    = 0.151 
            producer.avgTracksPerCell   = 0.130 
            producer.maxNumberOfDoublets = str(5*512*1024) # could be lowered to 1.4M, keeping the same for a fair comparison with master
            producer.maxNumberOfTuples   = str(256 * 1024) # could be lowered to 120k, same as above
      
   return process

