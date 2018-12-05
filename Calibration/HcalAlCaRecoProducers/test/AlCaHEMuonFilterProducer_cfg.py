import FWCore.ParameterSet.Config as cms

process = cms.Process("AlCaHEMuon")

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run2_mc']

process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")
process.load("Configuration.StandardSequences.AlCaRecoStreams_cff")
process.load("Configuration.EventContent.AlCaRecoOutput_cff")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
        'file:/eos/user/s/shghosh/FORSUNANDASIR/F256E27E-0930-E811-BE1A-FA163EBF1F42.root',
#       'root://cms-xrd-global.cern.ch///store/mc/RunIISpring18DR/DYToMuMu_M-20_13TeV_pythia8/GEN-SIM-RECO/NoPU_100X_upgrade2018_realistic_v10-v1/100000/F256E27E-0930-E811-BE1A-FA163EBF1F42.root'
        )
)

process.ALCARECOStreamHcalCalHEMuon = cms.OutputModule("PoolOutputModule",
                                                       SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalHEMuonFilter')
        ),
                                                       dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('ALCARECO'),
        filterName = cms.untracked.string('ALCARECOHcalCalHEMuon')
        ),
                                                       eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
                                                       outputCommands = process.OutALCARECOHcalCalHEMuonFilter.outputCommands,
                                                       fileName = cms.untracked.string('PoolOutput.root'),
                                                       )

# Path and EndPath definitions
process.endjob_step = cms.EndPath(process.endOfProcess)
process.ALCARECOStreamHcalCalHEMuonOutPath = cms.EndPath(process.ALCARECOStreamHcalCalHEMuon)

# Schedule definition
process.schedule = cms.Schedule(process.pathALCARECOHcalCalHEMuonFilter,process.endjob_step,process.ALCARECOStreamHcalCalHEMuonOutPath)
