# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: repr --processName=REPR --python_filename=reprocess_test_10_5_0_pre1.py --no_exec -s L1 --datatier GEN-SIM-DIGI-RAW -n 2 --era Phase2 --eventcontent FEVTDEBUGHLT --filein root://cms-xrd-global.cern.ch//store/mc/PhaseIIMTDTDRAutumn18DR/DYToLL_M-50_14TeV_pythia8/FEVT/PU200_pilot_103X_upgrade2023_realistic_v2_ext4-v1/280000/FF5C31D5-D96E-5E48-B97F-61A0E00DF5C4.root --conditions 103X_upgrade2023_realistic_v2 --beamspot HLLHC14TeV --geometry Extended2023D28 --fileout file:step2_2ev_reprocess_slim.root
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('REPR',eras.Phase2C8_trigger)
#process = cms.Process('REPR',eras.Phase2C4_timing_layer_bar)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023D41Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023D41_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

#process.Timing = cms.Service("Timing",
          #summaryOnly = cms.untracked.bool(False),
          #useJobReport = cms.untracked.bool(True)
#)

# Input source
process.source = cms.Source("PoolSource",
    #fileNames = cms.untracked.vstring('root://cms-xrd-global.cern.ch//store/mc/PhaseIIMTDTDRAutumn18DR/DYToLL_M-50_14TeV_pythia8/FEVT/PU200_pilot_103X_upgrade2023_realistic_v2_ext4-v1/280000/FF5C31D5-D96E-5E48-B97F-61A0E00DF5C4.root'),
    #fileNames = cms.untracked.vstring('root://eoscms/eos/cms/store/relval/CMSSW_10_6_0_pre3/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/105X_upgrade2023_realistic_v5_2023D41noPU-v1/10000/E6CBA1C6-7A2E-A540-97B3-DE2C30AB70C8.root'),
    #fileNames = cms.untracked.vstring('root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_6_0_pre3/RelValMuGunPt2To100/GEN-SIM-DIGI-RAW/105X_upgrade2023_realistic_v5_2023D41noPU-v2/10000/602E0B41-B698-6340-AC68-517578FEC457.root'),
    #fileNames = cms.untracked.vstring('root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_6_0_pre4/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU25ns_106X_upgrade2023_realistic_v2_2023D41PU200-v1/10000/FEA5D564-937A-8D4B-9C9A-696EFC05AB58.root'),
    #fileNames = cms.untracked.vstring('root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_6_0_patch2/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/106X_upgrade2023_realistic_v3_2023D41noPU-v1/10000/BC7B5A96-E3D2-ED48-81FC-35EF57134127.root'),
    fileNames = cms.untracked.vstring('root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring19DR/Mu_FlatPt2to100-pythia8-gun/GEN-SIM-DIGI-RAW/NoPU_106X_upgrade2023_realistic_v3-v1/60000/E0D5C6A5-B855-D14F-9124-0B2C9B28D0EA.root'),
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

# Schedule definition
#process.schedule = cms.Schedule(process.L1simulation_step,process.endjob_step,process.FEVTDEBUGHLToutput_step)
process.schedule = cms.Schedule(process.L1simulation_step,process.endjob_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)


# Customisation from command line

from L1Trigger.Configuration.customiseUtils import L1TrackTriggerTracklet,L1TTurnOffHGCalTPs_v9,configureCSCLCTAsRun2
process = L1TrackTriggerTracklet(process)
#process = L1TTurnOffHGCalTPs_v9(process)
process = configureCSCLCTAsRun2(process)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
