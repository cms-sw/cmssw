import FWCore.ParameterSet.Config as cms

#from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
#process = cms.Process("ANALYSIS",Run2_2018)
from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process("ANALYSIS",Run3)


process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
#process.GlobalTag.globaltag=autoCond['124X_dataRun3_v10']
process.GlobalTag.globaltag='140X_dataRun3_Prompt_v2'
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(1000)
if 'MessageLogger' in process.__dict__:
    process.MessageLogger.HcalIsoTrackX=dict()
    process.MessageLogger.HcalIsoTrack=dict()

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.load('Calibration.HcalCalibAlgos.hcalIsoTrackAnalyzer_cfi')
process.hcalIsoTrackAnalyzer.useRaw = 0   # 1 for Raw
process.hcalIsoTrackAnalyzer.debugEvents = [640818633, 640797426, 641251898,
                                            641261804, 641172007, 641031809]

process.source = cms.Source("PoolSource", 
                            fileNames = cms.untracked.vstring('/store/data/Run2024C/EGamma0/ALCARECO/HcalCalIsoTrkProducerFilter-PromptReco-v1/000/379/433/00000/455129b2-c385-42c6-ba07-31b69145cfba.root')
                            #/store/data/Run2023D/EGamma1/ALCARECO/HcalCalIsoTrkProducerFilter-PromptReco-v2/000/370/667/00000/027375a6-b66a-4909-a23c-dc08278265c5.root')
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.TFileService = cms.Service("TFileService",
   fileName = cms.string('output_newalca.root')
)

process.p = cms.Path(process.hcalIsoTrackAnalyzer)

