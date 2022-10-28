import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
process = cms.Process("RaddamMuon",Run2_2017)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")  
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("RecoJets.Configuration.CaloTowersES_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run2_data']

process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")
process.load("Calibration.HcalCalibAlgos.hcalHBHEMuon_cfi")

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.HBHEMuon=dict()

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
                                'file:/afs/cern.ch/work/a/amkaur/public/ForSunandaDa/AlcaProducer_codecheck/old/OutputHBHEMuon_old_2017.root'
                            )
)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("ValidationOld.root")
)

process.hcalHBHEMuon.useRaw = 0
process.hcalHBHEMuon.unCorrect = True
process.hcalHBHEMuon.getCharge = True
process.hcalHBHEMuon.ignoreHECorr = False
process.hcalHBHEMuon.maxDepth = 7
process.hcalHBHEMuon.verbosity = 0
process.hcalTopologyIdeal.MergePosition = False

process.p = cms.Path(process.hcalHBHEMuon)
