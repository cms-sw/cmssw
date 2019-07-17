# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step2 --python_filename=rerun_test.py --no_exec -s L1 --datatier GEN-SIM-DIGI-RAW -n 1 --era Phase2_trigger --eventcontent FEVTDEBUGHLT --filein step2_2ev_reprocess_slim.root --conditions 100X_upgrade2023_realistic_v1 --beamspot HLLHC14TeV --geometry Extended2023D17 --fileout file:step3_2ev_rerun-L1_slim.root --customise=L1Trigger/L1TNtuples/customiseL1Ntuple.L1NtupleRAWEMUGEN_MC --customise=L1Trigger/Configuration/customiseUtils.DropDepricatedProducts --customise=L1Trigger/Configuration/customiseUtils.L1TrackTriggerTracklet --customise=L1Trigger/Configuration/customiseUtils.DropOutputProducts --customise_commands process.FEVTDEBUGHLToutput.compressionLevel = cms.untracked.int32(2)
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('L1',eras.Phase2_trigger)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023D17Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
            'file:/eos/cms/store/group/dpg_trigger/comm_trigger/L1Trigger/Phase2/reprocess/SingleNeutrino/SingleNeutrino_PhaseIIFall17D-L1TPU200_L1rerun_v1/180921_062439/0000/step2_2ev_reprocess_slim_999.root'
#'/store/mc/PhaseIIFall17D/SingleNeutrino/GEN-SIM-DIGI-RAW/L1TnoPU_93X_upgrade2023_realistic_v5-v1/20000/FC4D23FD-2B9D-E811-A2D0-0025904E9012.root'
            ),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step2 nevts:1'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:step3_2ev_rerun-L1_slim.root'),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '100X_upgrade2023_realistic_v1', '')

process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff')
process.load('CalibCalorimetry.CaloTPG.CaloTPGTranscoder_cfi')
# Path and EndPath definitions
process.pL1TkPrimaryVertex = cms.Path(process.L1TkPrimaryVertex)
process.pL1TkElectrons = cms.Path(process.L1TkElectrons)
process.pL1TrackerHTMiss = cms.Path(process.L1TrackerHTMiss)
process.pL1TkPhotons = cms.Path(process.L1TkPhotons)
process.pL1TkGlbMuon = cms.Path(process.L1TkGlbMuons)
process.pL1TkMuon = cms.Path(process.L1TkMuons)
process.pL1TrkMET = cms.Path(process.L1TrackerEtMiss)
process.pL1TkTauFromCalo = cms.Path(process.L1TkTauFromCalo)
process.pL1TkCaloHTMissVtx = cms.Path(process.L1TkCaloHTMissVtx)
process.pL1TrackerJets = cms.Path(process.L1TrackerJets)
process.pL1TkCaloJets = cms.Path(process.L1TkCaloJets)
process.pL1TkIsoElectrons = cms.Path(process.L1TkIsoElectrons)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

# Schedule definition
process.schedule = cms.Schedule(process.L1simulation_step,process.endjob_step,process.FEVTDEBUGHLToutput_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

# customisation of the process.

# Automatic addition of the customisation function from L1Trigger.L1TNtuples.customiseL1Ntuple
from L1Trigger.L1TNtuples.customiseL1Ntuple import L1NtupleRAWEMUGEN_MC 

#call to customisation function L1NtupleRAWEMUGEN_MC imported from L1Trigger.L1TNtuples.customiseL1Ntuple
#process = L1NtupleRAWEMUGEN_MC(process)

# Automatic addition of the customisation function from L1Trigger.Configuration.customiseUtils
from L1Trigger.Configuration.customiseUtils import L1TTurnOffHGCalTPs,DropDepricatedProducts,L1TrackTriggerTracklet,DropOutputProducts

#call to customisation function L1TTurnOffHGCalTPs imported from L1Trigger.Configuration.customiseUtils
process = L1TTurnOffHGCalTPs(process)

#call to customisation function DropDepricatedProducts imported from L1Trigger.Configuration.customiseUtils
process = DropDepricatedProducts(process)

#call to customisation function L1TrackTriggerTracklet imported from L1Trigger.Configuration.customiseUtils
#process = L1TrackTriggerTracklet(process)

#call to customisation function DropOutputProducts imported from L1Trigger.Configuration.customiseUtils
process = DropOutputProducts(process)

# End of customisation functions

# Customisation from command line

process.FEVTDEBUGHLToutput.compressionLevel = cms.untracked.int32(2)
# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
