import FWCore.ParameterSet.Config as cms

process = cms.Process("Validation")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'

process.load("CalibMuon.DTCalibration.dt_offlineAnalysis_common_cosmics_cff")
process.load("DQMServices.Core.DQM_cfg")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.load("CalibMuon.DTCalibration.dtCalibValidation_cfi")
process.load("CalibMuon.DTCalibration.ALCARECODtCalibHLTDQM_cfi")
#process.ALCARECODtCalibHLTDQM.directory = "DT/HLTSummary"
#process.ALCARECODtCalibHLTDQM.eventSetupPathsKey = ''
#process.ALCARECODtCalibHLTDQM.HLTPaths = ['HLT_.*']

process.output = cms.OutputModule("PoolOutputModule",
                  outputCommands = cms.untracked.vstring('drop *', 
                                                         'keep *_MEtoEDMConverter_*_Validation'),
                  fileName = cms.untracked.string('DQM.root')
)

process.load("DQMServices.Components.MEtoEDMConverter_cff")

"""
process.dtValidSequence = cms.Sequence(process.muonDTDigis*
                                       process.dt1DRecHits*process.dt2DSegments*process.dt4DSegments+
                                       process.dtCalibValidation+process.ALCARECODtCalibHLTDQM)
"""
process.dtValidSequence = cms.Sequence(process.dt1DRecHits*process.dt2DSegments*process.dt4DSegments+
                                       process.dtCalibValidation)#+process.ALCARECODtCalibHLTDQM)
process.analysis_step = cms.Path(process.dtValidSequence*process.MEtoEDMConverter)
process.out_step = cms.EndPath(process.output)
#process.DQM.collectorHost = ''
