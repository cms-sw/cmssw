# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: repr --processName=REPR --python_filename=reprocess_test_10_5_0_pre1.py --no_exec -s L1 --datatier GEN-SIM-DIGI-RAW -n 2 --era Phase2 --eventcontent FEVTDEBUGHLT --filein root://cms-xrd-global.cern.ch//store/mc/PhaseIIMTDTDRAutumn18DR/DYToLL_M-50_14TeV_pythia8/FEVT/PU200_pilot_103X_upgrade2023_realistic_v2_ext4-v1/280000/FF5C31D5-D96E-5E48-B97F-61A0E00DF5C4.root --conditions 103X_upgrade2023_realistic_v2 --beamspot HLLHC14TeV --geometry Extended2023D28 --fileout file:step2_2ev_reprocess_slim.root
import FWCore.ParameterSet.Config as cms

from Configuration.ProcessModifiers.convertHGCalDigisSim_cff import convertHGCalDigisSim
from Configuration.StandardSequences.Eras import eras

process = cms.Process('REPR',eras.Phase2_trigger,convertHGCalDigisSim)
#process = cms.Process('REPR',eras.Phase2C4_timing_layer_bar)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023D17Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023D17_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_9_3_7/RelValDarkSUSY_14TeV/GEN-SIM-DIGI-RAW/93X_upgrade2023_realistic_v5_pLHE2023D17noPU_mH_125_mGmD_20_14TeV_cT_1k-v2/10000/FC792020-7D3B-E911-B43B-FA163E1C09AE.root'
#'/store/mc/PhaseIIFall17D/DisplacedSUSY_SmuonToMuNeutralino_M-500_CTau-1000_TuneCUETP8M1_14TeV-pythia8/GEN-SIM-DIGI-RAW/L1TPU200_93X_upgrade2023_realistic_v5-v1/30000/046F0E78-39A6-E811-B8E5-5065F38142E1.root'
#'/store/relval/CMSSW_9_3_7/RelValDisplacedMuon5pairs_Pt30To100_Dxy_0_100/GEN-SIM-DIGI-RAW/93X_upgrade2023_realistic_v5_2023D17noPU-v1/20000/5CFA03F2-7A2A-E911-8E5D-0025905A608E.root',
#'/store/relval/CMSSW_9_3_7/RelValDisplacedMuon5pairs_Pt30To100_Dxy_0_100/GEN-SIM-DIGI-RAW/PU25ns_93X_upgrade2023_realistic_v5_2023D17PU200-v2/20000/FE46EFEC-F22D-E911-A57B-0242AC130002.root'
#'/store/mc/PhaseIIMTDTDRAutumn18DR/NeutrinoGun_E_10GeV/FEVT/PU200_103X_upgrade2023_realistic_v2-v1/280000/EFFCC733-2B7F-C645-929D-505B1E0949D6.root'
#'/store/cmst3/group/hzz/gpetrucc/tmp/prod104X/ParticleGun_PU200/ParticleGun_PU200.batch2.job46.root'
#'/store/mc/PhaseIIFall17D/SingleNeutrino/GEN-SIM-DIGI-RAW/L1TPU200_93X_upgrade2023_realistic_v5-v1/80000/00157B11-405C-E811-89CA-0CC47AFB81B4.root'
#'/store/mc/PhaseIIFall17D/WJetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8/GEN-SIM-DIGI-RAW/L1TPU200_93X_upgrade2023_realistic_v5-v3/30000/FE752DF4-5359-E811-BD94-A0369FD0B242.root'
#'/store/mc/PhaseIIFall17D/TT_TuneCUETP8M2T4_14TeV-powheg-pythia8/GEN-SIM-DIGI-RAW/L1TPU200_93X_upgrade2023_realistic_v5-v2/30000/04998910-0755-E811-B1FC-EC0D9A822606.root'
#'/store/relval/CMSSW_9_3_7/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU25ns_93X_upgrade2023_realistic_v5_2023D17PU200-v1/10000/0A52EE7F-1E2D-E811-86EB-0242AC130002.root'
#'/store/relval/CMSSW_9_3_7/RelValNuGun/GEN-SIM-DIGI-RAW/PU25ns_93X_upgrade2023_realistic_v5_2023D17PU200-v1/10000/F0067730-182D-E811-BAB9-0242AC130002.root'
#'/store/relval/CMSSW_9_3_7/RelValZMM_14/GEN-SIM-DIGI-RAW/93X_upgrade2023_realistic_v5_2023D17noPU-v1/10000/10CD1685-012D-E811-8706-003048FFD734.root'
#'/store/relval/CMSSW_9_3_7/RelValNuGun/GEN-SIM-DIGI-RAW/PU25ns_93X_upgrade2023_realistic_v5_2023D17PU200-v1/10000/EAB9A5D0-142D-E811-8D3E-0242AC130002.root',
#'/store/relval/CMSSW_9_3_7/RelValNuGun/GEN-SIM-DIGI-RAW/PU25ns_93X_upgrade2023_realistic_v5_2023D17PU200-v1/10000/0EA28FD0-142D-E811-BC43-0242AC130002.root',
#'/store/relval/CMSSW_9_3_7/RelValNuGun/GEN-SIM-DIGI-RAW/PU25ns_93X_upgrade2023_realistic_v5_2023D17PU200-v1/10000/20E391D0-142D-E811-810A-0242AC130002.root',
#'/store/relval/CMSSW_9_3_7/RelValNuGun/GEN-SIM-DIGI-RAW/PU25ns_93X_upgrade2023_realistic_v5_2023D17PU200-v1/10000/A0B2A0CC-142D-E811-B908-0242AC130002.root',
#'/store/relval/CMSSW_9_3_7/RelValNuGun/GEN-SIM-DIGI-RAW/PU25ns_93X_upgrade2023_realistic_v5_2023D17PU200-v1/10000/A6409AD0-142D-E811-A31E-0242AC130002.root',
#'/store/relval/CMSSW_9_3_7/RelValNuGun/GEN-SIM-DIGI-RAW/PU25ns_93X_upgrade2023_realistic_v5_2023D17PU200-v1/10000/88EE04D1-142D-E811-8FEE-0242AC130002.root',
#'/store/relval/CMSSW_9_3_7/RelValNuGun/GEN-SIM-DIGI-RAW/PU25ns_93X_upgrade2023_realistic_v5_2023D17PU200-v1/10000/EA81D2D1-142D-E811-88BE-0242AC130002.root',
),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('repr nevts:2'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:step2_2ev_reprocess_slim.root'),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition


# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, '103X_upgrade2023_realistic_v2', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

process.GlobalTag = GlobalTag(process.GlobalTag, '103X_upgrade2023_realistic_v2', '') 

process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff')
process.load('CalibCalorimetry.CaloTPG.CaloTPGTranscoder_cfi')

# Path and EndPath definitions
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

# menu trees

process.load("L1Trigger.L1TNtuples.l1PhaseIITreeProducer_cfi")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('L1NtuplePhaseII.root')
)




# Schedule definition
process.schedule = cms.Schedule(process.L1simulation_step,#process.extraCollectionsMenuTree,
            process.runmenutree,process.endjob_step)#,process.FEVTDEBUGHLToutput_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

# Customisation from command line

# Automatic addition of the customisation function from L1Trigger.Configuration.customiseUtils
from L1Trigger.Configuration.customiseUtils import DropDepricatedProducts,L1TrackTriggerTracklet,DropOutputProducts 

#call to customisation function DropDepricatedProducts imported from L1Trigger.Configuration.customiseUtils
process = DropDepricatedProducts(process)

from L1Trigger.Configuration.customiseUtils import L1TrackTriggerTracklet
process = L1TrackTriggerTracklet(process)

process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string('ERROR'),
    default = cms.untracked.PSet(
        limit = cms.untracked.int32(10000)
    )
)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
