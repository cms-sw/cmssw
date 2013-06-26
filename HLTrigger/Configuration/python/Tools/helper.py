"""
Helper functions to extract the dictionary with
 - all EDFilters
 - all EDProducers
 - all EDAnalyzers
 - all modules
either from a dictionary (either a cms.Process.__dict__ or from the locals() inside a _cff.py fragment)
"""

import FWCore.ParameterSet.Config as cms

def findEDFilters(holder):
  if isinstance(holder, cms.Process):
    return process.filters_()
  else:
    return dict( (name, module) for name, module in holder.iteritems() if isinstance(module, cms.EDFilter) )


def findEDProducers(holder):
  if isinstance(holder, cms.Process):
    return process.producers_()
  else:
    return dict( (name, module) for name, module in holder.iteritems() if isinstance(module, cms.EDProducer) )


def findEDAnalyzers(holder):
  if isinstance(holder, cms.Process):
    return process.analyzers_()
  else:
    return dict( (name, module) for name, module in holder.iteritems() if isinstance(module, cms.EDAnalyzer) )


def findModules(holder):
  if isinstance(holder, cms.Process):
    modules = dict()
    modules.upate(process.analyzers_())
    modules.upate(process.producers_())
    modules.upate(process.filters_())
    return modules
  else:
    return dict( (name, module) for name, module in holder.iteritems() if isinstance(module, (cms.EDAnalyzer, _cms.EDProducer, _cms.EDFilter)) )


