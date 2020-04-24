import FWCore.ParameterSet.Config as cms

process = cms.Process("ANAL")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Calibration.IsolatedParticles.electronStudy_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run1_mc'] 

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
                            fileNames =cms.untracked.vstring("file:zeer_simevent.root")
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('zeer_estudy.root')
)

process.electronStudy.Verbosity = 0
process.p1 = cms.Path(process.electronStudy)

