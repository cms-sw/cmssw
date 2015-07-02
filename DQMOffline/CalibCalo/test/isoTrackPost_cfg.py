import FWCore.ParameterSet.Config as cms
import os

process = cms.Process("CONV")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1

# summary
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:dqm.root',
                                                              )
                            )

process.load("Configuration.StandardSequences.EDMtoMEAtRunEnd_cff")
process.dqmSaver.referenceHandling = cms.untracked.string('all')

cmssw_version = os.environ.get('CMSSW_VERSION','CMSSW_X_Y_Z')
process.dqmSaver.workflow = '/Test/Cal/DQM'

process.load("DQMOffline.CalibCalo.PostProcessorHcalIsoTrack_cfi")
process.PostProcessorHcalIsoTrack.saveToFile = True

process.p = cms.Path(process.EDMtoME * process.PostProcessorHcalIsoTrack * process.dqmSaver)
