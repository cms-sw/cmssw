import FWCore.ParameterSet.Config as cms

process = cms.Process("RaddamMuon")

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")
from Configuration.AlCa.autoCond_condDBv2 import autoCond
process.load("Configuration.Geometry.GeometryIdeal_cff")
process.GlobalTag.globaltag=autoCond['run2_mc']

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
        'root://xrootd.unl.edu//store/mc/Phys14DR/DYToMuMu_M-50_Tune4C_13TeV-pythia8/GEN-SIM-RECO/PU20bx25_tsg_castor_PHYS14_25_V1-v1/10000/184C1AC9-A775-E411-9196-002590200824.root'
        )
                            )
process.load("Calibration.HcalAlCaRecoProducers.alcahbhemuon_cfi")
process.load("Calibration.HcalAlCaRecoProducers.alcaHBHEMuonFilter_cfi")
process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalHBHEMuon_Output_cff")

process.muonOutput = cms.OutputModule("PoolOutputModule",
                                      outputCommands = process.OutALCARECOHcalHBHEMuon.outputCommands,
                                      fileName = cms.untracked.string('PoolOutput.root'),
                                      )

process.p = cms.Path(process.HBHEMuonProd * process.AlcaHBHEMuonFilter)
process.e = cms.EndPath(process.muonOutput)
