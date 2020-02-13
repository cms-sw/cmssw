import FWCore.ParameterSet.Config as cms

def widenRoads(process):
   if hasattr(process,'cscSegments'):
      process.cscSegments.algo_psets[4].algo_psets[0].enlarge = cms.bool(True)
      process.cscSegments.algo_psets[4].algo_psets[1].enlarge = cms.bool(True)
      process.cscSegments.algo_psets[4].algo_psets[2].enlarge = cms.bool(True)
      process.cscSegments.algo_psets[4].algo_psets[3].enlarge = cms.bool(True)
      process.cscSegments.algo_psets[4].algo_psets[4].enlarge = cms.bool(True)
      process.cscSegments.algo_psets[4].algo_psets[5].enlarge = cms.bool(True)

      return process

