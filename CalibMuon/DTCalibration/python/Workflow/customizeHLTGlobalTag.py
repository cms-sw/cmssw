import FWCore.ParameterSet.Config as cms

def customizeHLTGlobalTag(process):
    process.GlobalTag.connect   = 'frontier://FrontierProd/CMS_COND_31X_GLOBALTAG'
    process.GlobalTag.pfnPrefix = cms.untracked.string('frontier://FrontierProd/')

    return process
