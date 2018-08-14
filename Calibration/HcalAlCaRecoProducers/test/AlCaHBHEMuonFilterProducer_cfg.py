import FWCore.ParameterSet.Config as cms

process = cms.Process("AlCaHBHEMuon")

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.AlCaRecoStreams_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run2_mc']

process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
        'file:/afs/cern.ch/user/a/amkalsi/public/RecoFileForAlcaProducer.root'
#       'root://xrootd.unl.edu//store/mc/Phys14DR/DYToMuMu_M-50_Tune4C_13TeV-pythia8/GEN-SIM-RECO/PU20bx25_tsg_castor_PHYS14_25_V1-v1/10000/184C1AC9-A775-E411-9196-002590200824.root'
        )
)

process.ALCARECOStreamHcalCalHBHEMuon = cms.OutputModule("PoolOutputModule",
                                                         SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalHBHEMuonFilter')
        ),
                                                         dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('ALCARECO'),
        filterName = cms.untracked.string('ALCARECOHcalCalHBHEMuon')
        ),
                                                         eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
                                                         outputCommands = process.OutALCARECOHcalCalHBHEMuonFilter.outputCommands,
                                                         fileName = cms.untracked.string('PoolOutput.root'),
                                      )

# Path and EndPath definitions
process.endjob_step = cms.EndPath(process.endOfProcess)
process.ALCARECOStreamHcalCalHBHEMuonOutPath = cms.EndPath(process.ALCARECOStreamHcalCalHBHEMuon)

# Schedule definition
process.schedule = cms.Schedule(process.pathALCARECOHcalCalHBHEMuonFilter,process.endjob_step,process.ALCARECOStreamHcalCalHBHEMuonOutPath)
