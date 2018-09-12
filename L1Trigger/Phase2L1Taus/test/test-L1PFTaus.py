# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step2 --python_filename=rerun_step2_L1_onMCL1_FEVTHLTDEBUG.py --no_exec -s L1 --datatier GEN-SIM-DIGI-RAW -n 1 --era Phase2_timing --eventcontent FEVTDEBUGHLT --filein file:/afs/cern.ch/user/r/rekovic/release/CMSSW_9_3_2/src/step2_DIGI_PU200_10ev.root --conditions 93X_upgrade2023_realistic_v2 --beamspot HLLHC14TeV --geometry Extended2023D17 --fileout file:step2_ZEE_PU200_1ev_rerun-L1-L1Ntuple.root --customise=L1Trigger/L1TNtuples/customiseL1Ntuple.L1NtupleEMU
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
process.load('L1Trigger.TrackFindingTracklet.L1TrackletTracks_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(4000)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        #'root://cmsxrootd.fnal.gov///store/relval/CMSSW_9_3_7/RelValZTT_14TeV/MINIAODSIM/93X_upgrade2023_realistic_v5_2023D17noPU-v2/10000/C033607A-8E2C-E811-B4EF-0CC47A78A478.root'),
        'file:pickevents-1prong-pf-fail-mini.root'
        ),
    secondaryFileNames = cms.untracked.vstring(
        'file:pickevents-1prong-pf-fail-raw.root'
        #'root://cmsxrootd.fnal.gov///store/relval/CMSSW_9_3_7/RelValZTT_14TeV/GEN-SIM-DIGI-RAW/93X_upgrade2023_realistic_v5_2023D17noPU-v2/10000/02306B8E-6D2C-E811-9625-0025905B858C.root',
        #'root://cmsxrootd.fnal.gov///store/relval/CMSSW_9_3_7/RelValZTT_14TeV/GEN-SIM-DIGI-RAW/93X_upgrade2023_realistic_v5_2023D17noPU-v2/10000/163A5C8E-6D2C-E811-B990-0025905B85B2.root',
        #'root://cmsxrootd.fnal.gov///store/relval/CMSSW_9_3_7/RelValZTT_14TeV/GEN-SIM-DIGI-RAW/93X_upgrade2023_realistic_v5_2023D17noPU-v2/10000/1A7E6D8F-6D2C-E811-8DFB-0025905B85BE.root',
        #'root://cmsxrootd.fnal.gov///store/relval/CMSSW_9_3_7/RelValZTT_14TeV/GEN-SIM-DIGI-RAW/93X_upgrade2023_realistic_v5_2023D17noPU-v2/10000/5864268A-6D2C-E811-90F5-0025905A60EE.root',
        #'root://cmsxrootd.fnal.gov///store/relval/CMSSW_9_3_7/RelValZTT_14TeV/GEN-SIM-DIGI-RAW/93X_upgrade2023_realistic_v5_2023D17noPU-v2/10000/627E087F-6D2C-E811-8F81-0CC47A4C8E34.root',
        #'root://cmsxrootd.fnal.gov///store/relval/CMSSW_9_3_7/RelValZTT_14TeV/GEN-SIM-DIGI-RAW/93X_upgrade2023_realistic_v5_2023D17noPU-v2/10000/64AE2B7F-6D2C-E811-800E-0CC47A4D76A0.root',
        #'root://cmsxrootd.fnal.gov///store/relval/CMSSW_9_3_7/RelValZTT_14TeV/GEN-SIM-DIGI-RAW/93X_upgrade2023_realistic_v5_2023D17noPU-v2/10000/820C0278-6D2C-E811-B95B-0CC47A4C8F0C.root',
        #'root://cmsxrootd.fnal.gov///store/relval/CMSSW_9_3_7/RelValZTT_14TeV/GEN-SIM-DIGI-RAW/93X_upgrade2023_realistic_v5_2023D17noPU-v2/10000/BE6A3C80-6D2C-E811-9143-0CC47A4D7668.root',
        #'root://cmsxrootd.fnal.gov///store/relval/CMSSW_9_3_7/RelValZTT_14TeV/GEN-SIM-DIGI-RAW/93X_upgrade2023_realistic_v5_2023D17noPU-v2/10000/DA7C408A-6D2C-E811-9BFF-0025905B858A.root'
        ),
    inputCommands = cms.untracked.vstring("keep *", 
        "drop l1tHGCalTowerMapBXVector_hgcalTriggerPrimitiveDigiProducer_towerMap_HLT",
        "drop l1tEMTFHit2016Extras_simEmtfDigis_CSC_HLT",
        "drop l1tEMTFHit2016Extras_simEmtfDigis_RPC_HLT",
        "drop l1tEMTFHit2016s_simEmtfDigis__HLT",
        "drop l1tEMTFTrack2016Extras_simEmtfDigis__HLT",
        "drop l1tEMTFTrack2016s_simEmtfDigis__HLT")
     #skipEvents = cms.untracked.uint32(80)
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step2 nevts:1'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)


# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '100X_upgrade2023_realistic_v1', '')


process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff')
process.load('CalibCalorimetry.CaloTPG.CaloTPGTranscoder_cfi')

process.load('L1Trigger.L1THGCal.hgcalTriggerPrimitives_cff')
process.hgcl1tpg_step = cms.Path(process.hgcalTriggerPrimitives)

process.load('SimCalorimetry.EcalEBTrigPrimProducers.ecalEBTriggerPrimitiveDigis_cff')
process.EcalEBtp_step = cms.Path(process.simEcalEBTriggerPrimitiveDigis)

process.L1TrackTrigger_step = cms.Path(process.L1TrackletTracksWithAssociators)

process.VertexProducer.l1TracksInputTag = cms.InputTag("TTTracksFromTracklet", "Level1TTTracks")

# Path and EndPath definitions
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.endjob_step = cms.EndPath(process.endOfProcess)

process.load("L1Trigger.Phase2L1ParticleFlow.pfTracksFromL1Tracks_cfi")

from L1Trigger.Phase2L1ParticleFlow.l1ParticleFlow_cff import *
process.l1pf = cms.Path(process.pfTracksFromL1Tracks+process.l1ParticleFlow)

############################################################
# L1 Tau object
############################################################

process.load("L1Trigger.Phase2L1Taus.L1PFTauProducer_grow_l1t_cff")
process.L1PFTauProducer.debug = cms.untracked.bool(True)
process.L1PFTauProducer.L1PFObjects = cms.InputTag("l1pfProducer","PF")
process.L1PFTauProducer.L1Neutrals = cms.InputTag("l1pfProducer")
process.L1PFTaus = cms.Path(process.L1PFTauProducer)


# Schedule definition
process.schedule = cms.Schedule(process.EcalEBtp_step,process.L1TrackTrigger_step,process.L1simulation_step,process.l1pf,process.L1PFTaus,process.endjob_step)

from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion

#dump_file = open('dump.py','w')
#dump_file.write(process.dumpPython())
