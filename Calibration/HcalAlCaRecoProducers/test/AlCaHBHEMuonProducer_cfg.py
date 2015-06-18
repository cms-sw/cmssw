import FWCore.ParameterSet.Config as cms

process = cms.Process("AlCaHBHEMuon")

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run2_data']

process.load("Calibration.HcalAlCaRecoProducers.alcahbhemuon_cfi")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )
process.source = cms.Source("PoolSource",
                            # replace 'myfile.root' with the source file you want to use
                            fileNames = cms.untracked.vstring(
        'file:/afs/cern.ch/user/a/amkalsi/public/RecoFileForAlcaProducer.root'
#       'root://xrootd.unl.edu//store/mc/Phys14DR/DYToMuMu_M-50_Tune4C_13TeV-pythia8/GEN-SIM-RECO/PU20bx25_tsg_castor_PHYS14_25_V1-v1/10000/184C1AC9-A775-E411-9196-002590200824.root'
        )
)

process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalHBHEMuon_Output_cff")

process.muonOutput = cms.OutputModule("PoolOutputModule",
                                      outputCommands = process.OutALCARECOHcalHBHEMuon.outputCommands,
                                      fileName = cms.untracked.string('PoolOutput.root'),
                                      )

process.p = cms.Path(process.HBHEMuonProd)
process.e = cms.EndPath(process.muonOutput)
