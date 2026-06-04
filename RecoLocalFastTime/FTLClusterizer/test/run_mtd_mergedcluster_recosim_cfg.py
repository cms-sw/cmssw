import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
process = cms.Process("TEST", Phase2C17I13M9)

# Set up command line argument parsing
options = VarParsing('analysis')

# Define custom options
options.register('useSimTopologicalClustering',
                 True,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.bool,
                 "Enable topological clustering in SIM MergedCluster producer")

# Parse command line arguments
options.parseArguments()

# essential things
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.Geometry.GeometryExtendedRun4D110Reco_cff")
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T33', '')

process.MessageLogger.cerr.threshold = 'INFO'
# process.MessageLogger.cerr.threshold = 'DEBUG'

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:/eos/user/p/pakrap/MTD/CMSSW_15_0_0_pre2/src/Validation/MtdValidation/29706.0_SinglePiFlatPt0p7To10+Run4D110/1000evt/step3.root")    
)

# Load SimMergedCluster producer
from SimFastTiming.MtdSimMergedClusterProducers.mtdSimMergedClusterProducer_cfi import mtdSimMergedClusterProducer

process.mtdSimMergedClusterProducer = mtdSimMergedClusterProducer.clone(
    useTopologicalClustering = cms.bool(options.useSimTopologicalClustering)
)

process.mtdMergedClusters = cms.EDProducer("MTDMergedClusterProducer",
    btlBarrel = cms.InputTag("mtdClusters", "FTLBarrel"),
    etlEndcap = cms.InputTag("mtdClusters", "FTLEndcap"),
    btlMergedClusterInstance = cms.string("FTLBarrel"),
    etlMergedClusterInstance = cms.string("FTLEndcap"),
    timeThreshold = cms.double(10.0),
    energyThreshold = cms.double(1.0)
)

# Load MTD truth map associators (needed for the producer)
from SimFastTiming.MtdAssociatorProducers.mtdSimLayerClusterToTPAssociatorByTrackId_cfi import mtdSimLayerClusterToTPAssociatorByTrackId
from SimFastTiming.MtdAssociatorProducers.mtdSimLayerClusterToTPAssociation_cfi import mtdSimLayerClusterToTPAssociation

# clone and add label 
process.mtdSimLayerClusterToTPAssociatorByTrackId = mtdSimLayerClusterToTPAssociatorByTrackId.clone()
process.mtdSimLayerClusterToTPAssociation = mtdSimLayerClusterToTPAssociation.clone()

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(options.outputFile),
    outputCommands = cms.untracked.vstring(
        #'keep *'
        'drop *',

        "keep *_genParticles_*_*",  # keep GenParticles
        "keep *_mtdSimLayerClusterToTPAssociation_*_*",
        "keep *_mix_*_*",  # Keep mix to have TrackingParticles, MtdSimLayerClusters
        # 'keep *_mix_FTLBarrel_*',
        # 'keep *_mix_FTLEndcap_*',
        'keep *_mtdRecoClusterToSimLayerClusterAssociation_*_*',
        'keep *_mtdSimLayerClusterToRecoClusterAssociation_*_*',
        'keep *_mtdSimLayerClusterToTPAssociation_*_*',
        "keep *_mtdSimLayerClusterToTPAssociatorByTrackId_*_*",
        'keep *_mtdRecHits_FTLBarrel_*',
        'keep *_mtdRecHits_FTLEndcap_*',
        'keep *_mtdUncalibratedRecHits_FTLBarrel_*',
        'keep *_mtdUncalibratedRecHits_FTLEndcap_*',
        'keep *_trackExtenderWithMTD_generalTrackmtdpos_*',
        'keep *_trackExtenderWithMTD_generalTracksigmatmtd_*',
        'keep *_trackExtenderWithMTD_generalTracktmtd_*',
        'keep *_mtdTrackQualityMVA_mtdQualMVA_*',
        'keep *_mtdClusters_FTLBarrel_*',
        'keep *_mtdClusters_FTLEndcap_*',
        'keep *_mtdTrackingRecHits_*_*',
        'keep *_mtdMergedClusters_*_*',  
        "keep *_mtdSimMergedClusterProducer_*_*",
    ),
    #SelectEvents = cms.untracked.PSet(
    #    SelectEvents = cms.vstring('p')
    #)
)

process.mergedClusterSequence = cms.Sequence(process.mtdSimLayerClusterToTPAssociatorByTrackId + process.mtdSimLayerClusterToTPAssociation + process.mtdSimMergedClusterProducer + process.mtdMergedClusters)

process.p = cms.Path(process.mergedClusterSequence)
process.out_step = cms.EndPath(process.output)
process.schedule = cms.Schedule(process.p, process.out_step)

print("Testing BTL MTDMergedClusterProducer with adjacent cluster algorithm...")
print(f"Output file: {options.outputFile}")
print(f"Max events: {options.maxEvents}")
print(f"Use topological clustering (for SIM merged clusters): {options.useSimTopologicalClustering}")