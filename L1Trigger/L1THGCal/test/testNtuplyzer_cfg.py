import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('ntuple',eras.Phase2C2)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023D4Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023D4_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.VtxSmearedNoSmear_cff')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('L1Trigger.L1THGCal.hgcalTriggerPrimitives_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
    )

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
#    fileNames = cms.untracked.vstring('root://cms-xrd-global.cern.ch//store/mc/PhaseIIFall16DR82/MinBias_200PU_TuneCUETP8M1_14TeV-pythia8/GEN-SIM-RECO/PU200_90X_upgrade2023_realistic_v1-v1/60000/0047077C-B4ED-E611-8B36-FA163E78D122.root')
    fileNames = cms.untracked.vstring('root://cms-xrd-global.cern.ch//store/relval/CMSSW_9_0_0_pre4/RelValTTbar_14TeV/GEN-SIM-RECO/PU25ns_90X_upgrade2023_realistic_v3_D4TPU200c2-v1/10000/28524D8F-20F0-E611-B999-0CC47A7C354C.root')
)
# Additional output definition
process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("ntuple.root")
    )

# Path and EndPath definitions
process.digitisation_step = cms.Path(process.pdigi_valid)
process.L1simulation_step = cms.Path(process.SimL1Emulator)

# Remove best choice selection
process.hgcalTriggerPrimitiveDigiProducer.FECodec.NData = cms.uint32(999)
process.hgcalTriggerPrimitiveDigiProducer.FECodec.DataLength = cms.uint32(8)
process.hgcalTriggerPrimitiveDigiProducer.FECodec.triggerCellTruncationBits = cms.uint32(7)

process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms[0].calib_parameters.cellLSB = cms.double(
        process.hgcalTriggerPrimitiveDigiProducer.FECodec.linLSB.value() * 
        2 ** process.hgcalTriggerPrimitiveDigiProducer.FECodec.triggerCellTruncationBits.value() 
)

#chose the C2d-algorithm to run [NNC2d or dRC2d]
process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms[0].C2d_parameters.clusterType = cms.string('NNC2d')
# Adjust C2d thresholds
process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms[0].C2d_parameters.seeding_threshold = cms.double(5)
process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms[0].C2d_parameters.clustering_threshold = cms.double(2)
# Adjust the max-dR for geometric dRC2d-clustering
process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms[0].C2d_parameters.dR_cluster = cms.double(3.)
# Adjust the dR in the projected (x/z, y/z) plane
process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms[0].C3d_parameters.dR_multicluster = cms.double(0.01)
# Adjust the minimum pt required to produce a C3d
process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms[0].C3d_parameters.minPt_multicluster = cms.double(0.)

trgCells_algo_all =  cms.PSet( AlgorithmName = cms.string('SingleCellClusterAlgoBestChoice'),
                               FECodec = process.hgcalTriggerPrimitiveDigiProducer.FECodec,
                               HGCalEESensitive_tag = cms.string('HGCalEESensitive'),
                               HGCalHESiliconSensitive_tag = cms.string('HGCalHESiliconSensitive'),
                               calib_parameters = process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms[0].calib_parameters
                              )
cluster_algo_all =  cms.PSet( AlgorithmName = cms.string('HGCClusterAlgoBestChoice'),
                              FECodec = process.hgcalTriggerPrimitiveDigiProducer.FECodec,
                              HGCalEESensitive_tag = cms.string('HGCalEESensitive'),
                              HGCalHESiliconSensitive_tag = cms.string('HGCalHESiliconSensitive'),                           
                              calib_parameters = process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms[0].calib_parameters,
                              C2d_parameters = process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms[0].C2d_parameters,
                              C3d_parameters = process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms[0].C3d_parameters
                              )

process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms = cms.VPSet( cluster_algo_all )
process.hgcl1tpg_step = cms.Path( process.hgcalTriggerPrimitives ) 
process.digi2raw_step = cms.Path( process.DigiToRaw )

process.endjob_step = cms.EndPath(process.endOfProcess)

# load ntuplizer
process.load('L1Trigger.L1THGCal.hgcalTriggerNtuples_cff')
process.ntuple_step = cms.Path(process.hgcalTriggerNtuples) 
                                   
# Schedule definition
process.schedule = cms.Schedule( process.hgcl1tpg_step, 
                                 #process.digi2raw_step,
                                 process.ntuple_step, # create the persistent event 
                                 process.endjob_step
                                 )

# filter all path with the production filter sequence
for path in process.paths:
        getattr(process,path)._seq

