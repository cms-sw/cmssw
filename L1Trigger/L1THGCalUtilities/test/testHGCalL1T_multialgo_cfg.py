import FWCore.ParameterSet.Config as cms 
from Configuration.ProcessModifiers.convertHGCalDigisSim_cff import convertHGCalDigisSim

# For old samples use the digi converter
#from Configuration.Eras.Era_Phase2_cff import Phase2
#process = cms.Process('DIGI',Phase2,convertHGCalDigisSim)
process = cms.Process('DIGI',Phase2)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023D17Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023D17_cff')
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
       fileNames = cms.untracked.vstring('/store/relval/CMSSW_10_4_0_pre2/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/103X_upgrade2023_realistic_v2_2023D21noPU-v1/20000/F4344045-AEDE-4240-B7B1-27D2CF96C34E.root'),
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
import L1Trigger.L1THGCalUtilities.clustering2d as clustering2d
import L1Trigger.L1THGCalUtilities.clustering3d as clustering3d
import L1Trigger.L1THGCalUtilities.selectors as selectors
import L1Trigger.L1THGCalUtilities.customNtuples as ntuple


chains = HGCalTriggerChains()
# Register algorithms
## VFE
chains.register_vfe("Floatingpoint8", lambda p : vfe.create_compression(p, 4, 4, True))
## ECON
chains.register_concentrator("Supertriggercell", concentrator.create_supertriggercell)
chains.register_concentrator("Threshold", concentrator.create_threshold)
## BE1
chains.register_backend1("Ref2d", clustering2d.create_constrainedtopological)
chains.register_backend1("Dummy", clustering2d.create_dummy)
## BE2
chains.register_backend2("Ref3d", clustering3d.create_distance)
chains.register_backend2("Histomax", clustering3d.create_histoMax)
chains.register_backend2("Histomaxvardrth0", lambda p,i : clustering3d.create_histoMax_variableDr(p,i,seed_threshold=0.))
chains.register_backend2("Histomaxvardrth10", lambda p,i : clustering3d.create_histoMax_variableDr(p,i,seed_threshold=10.))
chains.register_backend2("Histomaxvardrth20", lambda p,i : clustering3d.create_histoMax_variableDr(p,i,seed_threshold=20.))
# Register selector
chains.register_selector("Genmatch", selectors.create_genmatch)
# Register ntuples
# Store gen info only in the reference ntuple
ntuple_list_ref = ['event', 'gen', 'multiclusters']
ntuple_list = ['event', 'multiclusters']
chains.register_ntuple("Genclustersntuple", lambda p,i : ntuple.create_ntuple(p,i, ntuple_list_ref))
chains.register_ntuple("Clustersntuple", lambda p,i : ntuple.create_ntuple(p,i, ntuple_list))

# Register trigger chains
## Reference chain
chains.register_chain('Floatingpoint8', "Threshold", 'Ref2d', 'Ref3d', 'Genmatch', 'Genclustersntuple')
concentrator_algos = ['Supertriggercell', 'Threshold']
backend_algos = ['Histomax', 'Histomaxvardrth0', 'Histomaxvardrth10', 'Histomaxvardrth20']
## Make cross product fo ECON and BE algos
import itertools
for cc,be in itertools.product(concentrator_algos,backend_algos):
    chains.register_chain('Floatingpoint8', cc, 'Dummy', be, 'Genmatch', 'Clustersntuple')

process = chains.create_sequences(process)

# Remove towers from sequence
process.hgcalTriggerPrimitives.remove(process.hgcalTowerMap)
process.hgcalTriggerPrimitives.remove(process.hgcalTower)

process.hgcl1tpg_step = cms.Path(process.hgcalTriggerPrimitives)
process.selector_step = cms.Path(process.hgcalTriggerSelector)
process.ntuple_step = cms.Path(process.hgcalTriggerNtuples)

# Schedule definition
process.schedule = cms.Schedule(process.hgcl1tpg_step, process.selector_step, process.ntuple_step)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion

