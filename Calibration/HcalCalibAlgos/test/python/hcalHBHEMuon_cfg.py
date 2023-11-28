import FWCore.ParameterSet.Config as cms

#from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
from Configuration.Eras.Era_Run3_DDD_cff import Run3_DDD
process = cms.Process("RaddamMuon",Run3_DDD)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")  
#process.load("Configuration.Geometry.GeometryExtended2023Reco_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("RecoJets.Configuration.CaloTowersES_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run3_data']

process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")
process.load("Calibration.HcalCalibAlgos.hcalHBHEMuon_cfi")

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.HBHEMuon=dict()

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
                                'file:/afs/cern.ch/work/h/hkaur/public/HCAL/June062023/002b36bd-33fd-4bac-b77c-0c918047ec98.root'
#                               'file:/afs/cern.ch/work/a/amkaur/public/ForSunandaDa/AlcaProducer_codecheck/old/OutputHBHEMuon_old_2017.root'
                            )
)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("ValidationOld.root")
)

process.hcalHBHEMuon.useRaw = 0
process.hcalHBHEMuon.unCorrect = False
process.hcalHBHEMuon.getCharge = True
process.hcalHBHEMuon.ignoreHECorr = False
process.hcalHBHEMuon.maxDepth = 7
process.hcalHBHEMuon.verbosity = 1111111
process.hcalHBHEMuon.pMinMuon = 10.0
process.hcalTopologyIdeal.MergePosition = False

process.p = cms.Path(process.hcalHBHEMuon)
