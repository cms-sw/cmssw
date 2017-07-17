import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'

process.load("CalibMuon.DTCalibration.dt_offlineAnalysis_common_cff")
process.GlobalTag.globaltag = ""

process.load("DQMServices.Core.DQM_cfg")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

#process.load("CalibMuon.DTCalibration.ALCARECODtCalib_cff")
process.load("DQM.DTMonitorModule.ALCARECODTCalibSynchDQM_cff")

process.output = cms.OutputModule("PoolOutputModule",
                  outputCommands = cms.untracked.vstring('drop *', 
                                                         'keep *_MEtoEDMConverter_*_DQM'),
                  fileName = cms.untracked.string('DQM.root')
)

process.load("DQMServices.Components.MEtoEDMConverter_cff")

process.dtLocalRecoSequence = cms.Sequence(process.dt1DRecHits*process.dt2DSegments*process.dt4DSegments)
process.ALCARECODTCalibSynchDQM_step = cms.Path(process.dtLocalRecoSequence+
                                                process.ALCARECODTCalibSynchDQM)
process.MEtoEDMConverter_step = cms.Path(process.MEtoEDMConverter)
process.out_step = cms.EndPath(process.output)
#process.DQM.collectorHost = ''
