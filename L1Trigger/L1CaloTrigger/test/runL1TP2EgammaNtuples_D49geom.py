import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
process = cms.Process('DIGI',eras.Phase2C9)


# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D49_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedHLLHC14TeV_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')


############################################################
# L1 tracking
############################################################

# remake stubs ?
process.load('L1Trigger.TrackTrigger.TrackTrigger_cff')
from L1Trigger.TrackTrigger.TTStubAlgorithmRegister_cfi import *
process.load("SimTracker.TrackTriggerAssociation.TrackTriggerAssociator_cff")
process.load("L1Trigger.TrackFindingTracklet.Tracklet_cfi")
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

# process.TTClusterStub = cms.Path(process.TrackTriggerClustersStubs)
# process.TTClusterStubTruth = cms.Path(process.TrackTriggerAssociatorClustersStubs)


process.TTTrackAssociatorFromPixelDigis.TTTracks = cms.VInputTag(
    cms.InputTag('TTTracksFromTrackletEmulation', 'Level1TTTracks'))

# emulation
process.TTTracksEmulationWithTruth = cms.Path(
    process.offlineBeamSpot *
    process.TTTracksFromTrackletEmulation *
    process.TrackTriggerAssociatorTracks)
# L1TRK_PROC.asciiFileName = cms.untracked.string("evlist.txt")





process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)

# Input source
process.source = cms.Source("PoolSource",
       fileNames = cms.untracked.vstring('file:/data/cerminar/Phase2HLTTDRWinter20DIGI/SingleElectron_PT2to200/GEN-SIM-DIGI-RAW/PU200_110X_mcRun4_realistic_v3_ext2-v2/F32C5A21-F0E9-9149-B04A-883CC704E820.root'),
       # fileNames = cms.untracked.vstring('/store/mc/PhaseIIMTDTDRAutumn18DR/SinglePion_FlatPt-2to100/FEVT/PU200_103X_upgrade2023_realistic_v2-v1/70000/FFA969EE-22E0-E447-86AA-46A6CBF6407D.root'),
       inputCommands=cms.untracked.vstring(
           'keep *',
           'drop l1tEMTFHit2016Extras_simEmtfDigis_CSC_HLT',
           'drop l1tEMTFHit2016Extras_simEmtfDigis_RPC_HLT',
           'drop l1tEMTFHit2016s_simEmtfDigis__HLT',
           'drop l1tEMTFTrack2016Extras_simEmtfDigis__HLT',
           'drop l1tEMTFTrack2016s_simEmtfDigis__HLT',
           'drop FTLClusteredmNewDetSetVector_mtdClusters_FTLBarrel_RECO',
           'drop FTLClusteredmNewDetSetVector_mtdClusters_FTLEndcap_RECO',
           'drop MTDTrackingRecHitedmNewDetSetVector_mtdTrackingRecHits__RECO',
           'drop BTLDetIdBTLSampleFTLDataFrameTsSorted_mix_FTLBarrel_HLT',
           'drop ETLDetIdETLSampleFTLDataFrameTsSorted_mix_FTLEndcap_HLT',
           )
       )

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.20 $'),
    annotation = cms.untracked.string('SingleElectronPt10_cfi nevts:10'),
    name = cms.untracked.string('Applications')
)

# Output definition
process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("ntuple.root")
    )

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

# load HGCAL TPG simulation
process.load('L1Trigger.L1THGCal.hgcalTriggerPrimitives_cff')

# To add truth-matched calo cells and downstream objects
# process.load('L1Trigger.L1THGCalUtilities.caloTruthCells_cff')
# process.hgcalTriggerPrimitives += process.caloTruthCells
# process.load('L1Trigger.L1THGCalUtilities.caloTruthCellsNtuples_cff')

process.hgcl1tpg_step = cms.Path(process.hgcalTriggerPrimitives)

# load Standalone EG producers
process.load('L1Trigger.L1CaloTrigger.l1EgammaStaProducers_cff')
process.l1EgammaStaProducers_step = cms.Path(process.l1EgammaStaProducers)

# load track matching modules
process.load('L1Trigger.L1TTrackMatch.L1TkEgammaObjects_cff')
process.l1EgammaTrackMatchProducers_step = cms.Path(process.l1TkElectronTrackEllipticProducers)

# load ntuplizer
process.load('L1Trigger.L1CaloTrigger.L1TCaloTriggerNtuples_cff')
process.ntuple_step = cms.Path(process.l1CaloTriggerNtuples)

# customization from Giovanni
# process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters.histoMax_C3d_seeding_parameters.threshold_histo_multicluster = 0.5
# process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters.histoMax_C3d_seeding_parameters.binSumsHisto = cms.vuint32(
#          3,  3,  3,  3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
#     )

# Schedule definition
process.schedule = cms.Schedule(
    process.TTTracksEmulationWithTruth,
    process.hgcl1tpg_step,
    process.l1EgammaTrackMatchProducers_step,
    process.l1EgammaStaProducers_step,
    process.ntuple_step)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
