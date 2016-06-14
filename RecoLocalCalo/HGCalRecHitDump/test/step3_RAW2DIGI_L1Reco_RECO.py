# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step3 --conditions auto:run2_mc -n 10 --era Phase2 --eventcontent FEVTDEBUGHLT -s RAW2DIGI,L1Reco,RECO:localreco --datatier GEN-SIM-RECO --customise SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023LReco --geometry Extended2023LReco -n 100 --no_exec --filein file:step2.root --fileout file:step3.root
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('RECO',eras.Phase2)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023LRecoReco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:step2.root'),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step3 nevts:100'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RECO'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(10485760),
    fileName = cms.untracked.string('file:step3.root'),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.TestHGCClusters = cms.EDProducer(
    'HGCalClusterTestProducer'
    )

process.load('CommonTools.UtilAlgos.TFileService_cfi')
process.TFileService.fileName = cms.string('hydra.root')

process.load('RecoParticleFlow.PFClusterProducer.particleFlowRecHitHGC_cff')

process.Hydra = cms.EDProducer(
    'HydraProducer',
    HGCRecHitCollection=cms.VInputTag("particleFlowRecHitHGC"),
    HGCalUncalibRecHitCollection = cms.VInputTag('HGCalUncalibRecHit:HGCEEUncalibRecHits',
                                                 'HGCalUncalibRecHit:HGCHEFUncalibRecHits'
                                                 ),
    GenParticleCollection=cms.InputTag("genParticles"),
    RecTrackCollection=cms.InputTag("generalTracks"),
    SimTrackCollection=cms.InputTag("g4SimHits"),
    SimVertexCollection=cms.InputTag("g4SimHits"),
    SimHitCollection = cms.VInputTag('g4SimHits:HGCHitsEE',
                                     'g4SimHits:HGCHitsHEfront')
    )
process.FakeClusterGen = cms.EDProducer(
    "HydraFakeClusterBuilder",HydraTag=cms.InputTag("Hydra"),
    SplitRecHits=cms.bool(False),
    UseGenParticles=cms.bool(True),
    MinDebugEnergy=cms.untracked.double(30.)
    )
process.FakeClusterCaloFace = cms.EDProducer(
    "HydraFakeClusterBuilder",HydraTag=cms.InputTag("Hydra"),
    SplitRecHits=cms.bool(False),
    UseGenParticles=cms.bool(False),
    MinDebugEnergy=cms.untracked.double(30.)
    )

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.localreco)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.endjob_step,process.FEVTDEBUGHLToutput_step)

# customisation of the process.

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.combinedCustoms
from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2023LReco 

#call to customisation function cust_2023LReco imported from SLHCUpgradeSimulations.Configuration.combinedCustoms
process = cust_2023LReco(process)

# End of customisation functions
process.localreco += process.particleFlowRecHitHGCSeq
process.localreco += process.Hydra
process.localreco += process.FakeClusterGen
process.localreco += process.FakeClusterCaloFace
process.localreco += process.TestHGCClusters
