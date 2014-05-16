# Auto generated configuration file
# using: 
# Revision: 1.20 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step3 --conditions auto:upgrade2019 -n 10 --eventcontent FEVTDEBUGHLT,DQM -s RAW2DIGI,L1Reco,RECO,VALIDATION,DQM --datatier GEN-SIM-RECO,DQM --customise SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019 --geometry Extended2019 --magField 38T_PostLS1 --filein file:step2.root --fileout file:step3.root
import FWCore.ParameterSet.Config as cms

process = cms.Process('RERECO')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_6_2_0_SLHC12/RelValZTT_14TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v1/00000/70C52109-A4D4-E311-9E86-002354EF3BD2.root'

    )
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.20 $'),
    annotation = cms.untracked.string('step3 nevts:10'),
    name = cms.untracked.string('Applications')
)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    fileName = cms.untracked.string('file:reco.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-RECO')
    )
)


# Other statements
process.mix.digitizers = cms.PSet()
for a in process.aliases: delattr(process, a)
process.RandomNumberGeneratorService.restoreStateLabel=cms.untracked.string("randomEngineStateProducer")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')


process.load("RecoParticleFlow.PFClusterProducer.particleFlowRecHitHF_cfi")
process.load("RecoParticleFlow.PFClusterProducer.particleFlowClusterHF_cfi")
process.load("RecoParticleFlow.PFClusterProducer.particleFlowRecHitHBHEHO_cfi")
process.load("RecoParticleFlow.PFTracking.particleFlowTrack_cff")
process.load("RecoParticleFlow.PFProducer.particleFlowSimParticle_cff")

process.hcalSeeds = cms.EDProducer('PFSeedSelector',
                                      src = cms.InputTag('particleFlowRecHitHBHEHO')
)                                      


# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.pfTrack+process.particleFlowCluster+process.particleFlowRecHitHF+process.particleFlowClusterHF+process.particleFlowRecHitHBHEHO+process.hcalSeeds+process.particleFlowSimParticle)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)


# Schedule definition
process.schedule = cms.Schedule(process.reconstruction_step,process.FEVTDEBUGHLToutput_step)

# customisation of the process.

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.combinedCustoms
from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2019 

#call to customisation function cust_2019 imported from SLHCUpgradeSimulations.Configuration.combinedCustoms
process = cust_2019(process)

# Automatic addition of the customisation function from SimGeneral.MixingModule.fullMixCustomize_cff
#from SimGeneral.MixingModule.fullMixCustomize_cff import setCrossingFrameOn 

#call to customisation function setCrossingFrameOn imported from SimGeneral.MixingModule.fullMixCustomize_cff
#process = setCrossingFrameOn(process)
process.pfTrack.MuColl = cms.InputTag('muons')
# End of customisation functions

