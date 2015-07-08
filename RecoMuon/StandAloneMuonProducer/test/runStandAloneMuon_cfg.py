import FWCore.ParameterSet.Config as cms
#process = cms.Process("RecoSTAMuon")
process = cms.Process("STARECO")
process.load("RecoMuon.Configuration.RecoMuon_cff")
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
#process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
#process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.Geometry.GeometryExtended2023Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff') #!!!!!!!!!!!!!!!!!!!!!!!!!!
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_design', '')

# Fix DT and CSC Alignment #
############################
#from SLHCUpgradeSimulations.Configuration.fixMissingUpgradeGTPayloads import fixDTAlignmentConditions
#process = fixDTAlignmentConditions(process)
#from SLHCUpgradeSimulations.Configuration.fixMissingUpgradeGTPayloads import fixCSCAlignmentConditions
#process = fixCSCAlignmentConditions(process)

process.maxEvents = cms.untracked.PSet(
input = cms.untracked.int32(-1)
)

 # Seed generator
from RecoMuon.MuonSeedGenerator.standAloneMuonSeeds_cff import *

# Stand alone muon track producer
from RecoMuon.StandAloneMuonProducer.standAloneMuons_cff import *

# Beam Spot 
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *

process.source = cms.Source("PoolSource",
fileNames = cms.untracked.vstring(

                 )
           )

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(
        'file:/tmp/archie/out_STA_reco_Pt200_withGems_new.root'
    ),
    outputCommands = cms.untracked.vstring(
        'keep  *_*_*_*',
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('stareco_step')
    )

 )


process.stareco_step = cms.Path(offlineBeamSpot*standAloneMuonSeeds*process.standAloneMuons)
process.endjob_step  = cms.Path(process.endOfProcess)
process.out_step     = cms.EndPath(process.output)

process.schedule = cms.Schedule(
    process.stareco_step,
    process.endjob_step,
    process.out_step
)

