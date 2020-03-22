import FWCore.ParameterSet.Config as cms

def widenRoads(process):
   if hasattr(process,'cscSegments'):
      for ps in process.cscSegments.algo_psets[4].algo_psets: ps.enlarge = True   
      return process
