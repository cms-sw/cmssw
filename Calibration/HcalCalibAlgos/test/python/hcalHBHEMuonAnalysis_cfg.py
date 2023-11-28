import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
#from Configuration.Eras.Era_Run3_DDD_cff import Run3_DDD
process = cms.Process("RaddamMuon",Run2_2017)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")  
#process.load("Configuration.Geometry.GeometryExtended2017Reco_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run2_data']
#process.GlobalTag.globaltag=autoCond['run3_data']

process.load("Calibration.HcalCalibAlgos.hcalHBHEMuonAnalysis_cfi")

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.HBHEMuon=dict()

process.maxEvents = cms.untracked.PSet( 
    input = cms.untracked.int32(-1) 
)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
                                'file:/afs/cern.ch/work/a/amkaur/public/ForSunandaDa/AlcaProducer_codecheck/new/OutputBHEMuonProducerFilter_new_2017.root'
                            )
                    )

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("ValidationNew.root")
)

process.p = cms.Path(process.hcalHBHEMuonAnalysis)
