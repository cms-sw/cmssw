import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_2025_cff import Run3_2025
process = cms.Process("ANALYSIS",Run3_2025)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['phase1_2025_realistic']

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.HcalIsoTrack=dict()
process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.load('Calibration.HcalCalibAlgos.hcalIsoTrackAnalyzer_cfi')
process.source = cms.Source("PoolSource", 
                            fileNames = cms.untracked.vstring(
       'file:newPoolOutput.root',
    )
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.TFileService = cms.Service("TFileService",
   fileName = cms.string('outputNew.root')
)

process.hcalIsoTrackAnalyzer.fillInRange = True  # fils only 40-60 GeV
process.hcalIsoTrackAnalyzer.useRaw = 0   # 1 for Raw
process.hcalIsoTrackAnalyzer.unCorrect = 1   # 1 for RespCorr; 2 for Gain
process.hcalIsoTrackAnalyzer.dataType = 0   # 1 for from Jet; 0 for others

process.p = cms.Path(process.hcalIsoTrackAnalyzer)

