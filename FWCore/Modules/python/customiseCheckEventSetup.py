from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import six

def customise(process):
    process.escontent = cms.EDAnalyzer("PrintEventSetupContent",
        compact = cms.untracked.bool(True),
        printProviders = cms.untracked.bool(True)
    )
    process.esretrieval = cms.EDAnalyzer("PrintEventSetupDataRetrieval",
        printProviders = cms.untracked.bool(True)
    )
    process.esout = cms.EndPath(process.escontent + process.esretrieval)

    if process.schedule_() is not None:
        process.schedule_().append(process.esout)

    for name, module in six.iteritems(process.es_sources_()):
        print("ESModules> provider:%s '%s'" % (name, module.type_()))
    for name, module in six.iteritems(process.es_producers_()):
        print("ESModules> provider:%s '%s'" % (name, module.type_()))

    return process
