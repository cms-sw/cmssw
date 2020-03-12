import FWCore.ParameterSet.Config as cms 
from Configuration.StandardSequences.Eras import eras
from Configuration.ProcessModifiers.convertHGCalDigisSim_cff import convertHGCalDigisSim

# For old samples use the digi converter
process = cms.Process('DIGI',eras.Phase2C4)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D35Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D35_cff')
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


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50)
)

# Input source
process.source = cms.Source("PoolSource",
       fileNames = cms.untracked.vstring('/store/mc/PhaseIIMTDTDRAutumn18DR/SinglePion_FlatPt-2to100/FEVT/NoPU_103X_upgrade2023_realistic_v2-v1/70000/60B619EA-9EC7-A744-B3E5-5369820F26CB.root'),
       inputCommands=cms.untracked.vstring(
           'keep *',
           'drop l1tEMTFHit2016Extras_simEmtfDigis_CSC_HLT',
           'drop l1tEMTFHit2016Extras_simEmtfDigis_RPC_HLT',
           'drop l1tEMTFHit2016s_simEmtfDigis__HLT',
           'drop l1tEMTFTrack2016Extras_simEmtfDigis__HLT',
           'drop l1tEMTFTrack2016s_simEmtfDigis__HLT',
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

process.load('L1Trigger.L1THGCalUtilities.HGC3DClusterGenMatchSelector_cff')
process.load('L1Trigger.L1THGCalUtilities.hgcalTriggerNtuples_cff')
from L1Trigger.L1THGCalUtilities.hgcalTriggerChains import HGCalTriggerChains
import L1Trigger.L1THGCalUtilities.vfe as vfe
import L1Trigger.L1THGCalUtilities.concentrator as concentrator
import L1Trigger.L1THGCalUtilities.truth_prod as truth_prod
import L1Trigger.L1THGCalUtilities.clustering2d as clustering2d
import L1Trigger.L1THGCalUtilities.clustering3d as clustering3d
import L1Trigger.L1THGCalUtilities.selectors as selectors
import L1Trigger.L1THGCalUtilities.customNtuples as ntuple

chains = HGCalTriggerChains()
# Register algorithms
## VFE
chains.register_vfe("Floatingpoint8", lambda p : vfe.create_compression(p, 4, 4, True))
## TRUTH PRODUCER
chains.register_truth_prod("TruthProd", truth_prod.create_truth_prod)
## BE1
chains.register_backend1("TruthDummy", clustering2d.create_truth_dummy)
## BE2
chains.register_backend2("Histomax", clustering3d.create_histoMax)
chains.register_backend2("Histomaxvardrth0", lambda p,i : clustering3d.create_histoMax_variableDr(p,i,seed_threshold=0.))
chains.register_backend2("Histomaxvardrth10", lambda p,i : clustering3d.create_histoMax_variableDr(p,i,seed_threshold=10.))
chains.register_backend2("Histomaxvardrth20", lambda p,i : clustering3d.create_histoMax_variableDr(p,i,seed_threshold=20.))
# Register selector
chains.register_selector("Genmatch", selectors.create_genmatch)
# Register ntuples
ntuple_list = ['event', 'gen', 'multiclusters']
chains.register_ntuple("Genclustersntuple", lambda p,i : ntuple.create_ntuple(p,i, ntuple_list))

# Register trigger chains
backend_algos = ['Histomax', 'Histomaxvardrth0', 'Histomaxvardrth10', 'Histomaxvardrth20']

chains.register_truth_chain('Floatingpoint8', 'TruthProd', '', '', '', 'Genclustersntuple')
for be in backend_algos:
    chains.register_truth_chain('Floatingpoint8', 'TruthProd', 'TruthDummy', be, 'Genmatch', 'Genclustersntuple')

process = chains.create_truth_sequences(process)

# Remove towers from sequence
process.hgcalTriggerPrimitivesTruth.remove(process.hgcalTowerMap)
process.hgcalTriggerPrimitivesTruth.remove(process.hgcalTower)

process.hgcl1tpg_truth_step = cms.Path(process.hgcalTriggerPrimitivesTruth)
process.selector_step = cms.Path(process.hgcalTriggerSelector)
process.ntuple_step = cms.Path(process.hgcalTriggerNtuples)

# Schedule definition
process.schedule = cms.Schedule(process.hgcl1tpg_truth_step, process.selector_step, process.ntuple_step)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion

