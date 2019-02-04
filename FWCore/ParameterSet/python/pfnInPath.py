import FWCore.ParameterSet.Config as cms
import os, os.path

def pfnInPath(name):
  for path in os.environ['CMSSW_SEARCH_PATH'].split(':'):
    fn = os.path.join(path, name)
    if os.path.isfile(fn):
      return 'file:' + fn

  raise IOError("No such file or directory '%s' in the CMSSW_SEARCH_PATH" % name)


cms.pfnInPath            = lambda name: cms.string(pfnInPath(name))
cms.untracked.pfnInPath  = lambda name: cms.untracked.string(pfnInPath(name))
cms.pfnInPaths           = lambda *names: cms.vstring(pfnInPath(name) for name in names)
cms.untracked.pfnInPaths = lambda *names: cms.untracked.vstring(pfnInPath(name) for name in names)
