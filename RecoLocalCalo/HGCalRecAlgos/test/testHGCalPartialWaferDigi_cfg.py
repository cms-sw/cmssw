import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9

process = cms.Process("TestPartial",Phase2C22I13M9)
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.Geometry.GeometryExtendedRun4D121Reco_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('RecoLocalCalo.HGCalRecAlgos.hgcalRecHitToolsPartialWafer_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T33', '')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalSim=dict()

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:hgcV17Digi.root') )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.analysis_step = cms.Path(process.hgcalCheckToolDigiEE+process.hgcalCheckToolDigiHE)

# Schedule definition
process.schedule = cms.Schedule(process.analysis_step)
