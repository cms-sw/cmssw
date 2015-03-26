import FWCore.ParameterSet.Config as cms

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

    for name, module in process.es_sources_().iteritems():
        print "ESModules> provider:%s '%s'" % (name, module.type_())
    for name, module in process.es_producers_().iteritems():
        print "ESModules> provider:%s '%s'" % (name, module.type_())

    return process
