# used by cmsDriver when called like 
#   cmsDiver.py hlt -s HLT:cached:<runnumber> 
# or
#   cmsDiver.py hlt -s HLT:cached:fromSource

hltByRun = {
  ( 190456, 193621 ) : '5E33v4',
  ( 193834, 196531 ) : '7E33v2',
  ( 198022, 199608 ) : '7E33v3',
  ( 199698, 202504 ) : '7E33v4',
  ( 202970, 203709 ) : '7E33v4',
  ( 203777, 208686 ) : '8E33v2',
}


def getCachedHLT(run) :
  # make sure run is an integer
  run = int(run)

  # look for a run range that contains the ginven run, and return the associated HLT key
  for key in hltByRun:
    if key[0] <= run <= key[1]:
      return hltByRun[key]

  # the given run was not found in any supported run range
  raise Exception('The given run number (%d) is not supported' % run)


def loadCachedHltConfiguration(process, run, fastsim = False):
  if fastsim:
    process.load('HLTrigger/Configuration/HLT_%s_Famos_cff' % getCachedHLT(run))
  else:
    process.load('HLTrigger/Configuration/HLT_%s_cff'       % getCachedHLT(run))

import FWCore.ParameterSet.Config as cms
cms.Process.loadCachedHltConfiguration = loadCachedHltConfiguration
