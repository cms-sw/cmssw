import FWCore.ParameterSet.Config as cms

process = cms.Process("ALCAISOTRACK")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run2_mc']

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
        'file:PoolOutput.root'
#       '/store/relval/CMSSW_7_4_0_pre6/RelValPhotonJets_Pt_10_13/GEN-SIM-RECO/MCRUN2_74_V1-v1/00000/6EC8FCC8-E2A8-E411-9506-002590596468.root'
 )
)

process.load("Calibration.HcalAlCaRecoProducers.alcaisotrk_cfi")
process.IsoProd.ProcessName = 'HLTNew1'

process.load("Calibration.HcalAlCaRecoProducers.alcastreamHcalIsotrkOutput_cff")
process.IsoTrackOutput = cms.OutputModule("PoolOutputModule",
   outputCommands = process.alcastreamHcalIsotrkOutput.outputCommands,
   fileName = cms.untracked.string('isotrack.root')
)

process.p = cms.Path(process.IsoProd)
process.e = cms.EndPath(process.IsoTrackOutput)
