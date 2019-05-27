# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step2 --python_filename=rerun_step2_L1_onMCL1_FEVTHLTDEBUG.py --no_exec -s L1 --datatier GEN-SIM-DIGI-RAW -n 1 --era Phase2_timing --eventcontent FEVTDEBUGHLT --filein file:/afs/cern.ch/user/r/rekovic/release/CMSSW_9_3_2/src/step2_DIGI_PU200_10ev.root --conditions 93X_upgrade2023_realistic_v2 --beamspot HLLHC14TeV --geometry Extended2023D17 --fileout file:step2_ZEE_PU200_1ev_rerun-L1-L1Ntuple.root --customise=L1Trigger/L1TNtuples/customiseL1Ntuple.L1NtupleEMU
import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

process = cms.Process("IN2", eras.Phase2C4_trigger)
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.Geometry.GeometryExtended2023D35Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023D35_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff')
process.load('CalibCalorimetry.CaloTPG.CaloTPGTranscoder_cfi')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('L1Trigger.TrackFindingTracklet.L1TrackletTracks_cff')
process.VertexProducer.l1TracksInputTag = cms.InputTag("TTTracksFromTracklet", "Level1TTTracks")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '103X_upgrade2023_realistic_v2', '') 

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
                            duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
                            fileNames = cms.untracked.vstring(
       *( 
            '/store/cmst3/group/monojet/trigger/GGH/inputs_753710142.root',
            '/store/cmst3/group/monojet/trigger/GGH/inputs_753710413.root') ),
                            inputCommands = cms.untracked.vstring("keep *", 
                                                                  "drop l1tHGCalTowerMapBXVector_hgcalTriggerPrimitiveDigiProducer_towerMap_HLT",
                                                                  "drop *_hgcalTriggerPrimitiveDigiProducer_*_*",
                                                                  "drop l1tEMTFHit2016Extras_simEmtfDigis_CSC_HLT",
                                                                  "drop l1tEMTFHit2016Extras_simEmtfDigis_RPC_HLT",
                                                                  "drop l1tEMTFHit2016s_simEmtfDigis__HLT",
                                                                  "drop l1tEMTFTrack2016Extras_simEmtfDigis__HLT",
                                                                  "drop l1tEMTFTrack2016s_simEmtfDigis__HLT")
                            )
process.options = cms.untracked.PSet(
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step2 nevts:1'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)
process.load("L1Trigger.Phase2L1Taus.L1NNTauProducer_cff")
process.L1NNTauProducer.L1PFObjects = cms.InputTag("l1pfCandidates","PF")
process.load("L1Trigger.Phase2L1Taus.L1PFTauProducer_cff")
process.L1PFTauProducer.L1PFObjects = cms.InputTag("l1pfCandidates","PF")
process.L1NNTauProducerPuppi = process.L1NNTauProducer.clone()
process.L1NNTauProducerPuppi.L1PFObjects = cms.InputTag("l1pfCandidates","Puppi")
process.L1NNTauProducerPuppi.NNFileName  = cms.string("L1Trigger/Phase2L1Taus/data/tau_3layer_puppi.pb")

process.p = cms.Path(
    #process.L1TrackletTracks +
    #process.SimL1Emulator + 
    process.L1PFTauProducer  +
    process.L1NNTauProducer  +
    process.L1PFTauProducerPuppi
)
process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string("tau104X.root"),
                               outputCommands = cms.untracked.vstring(
        'keep *_*_*_IN2',
        #"keep *_*_L1PFTau*_*",
        "keep *_genParticles_*_*",
        ),
                               )
process.e = cms.EndPath(process.out)
process.schedule = cms.Schedule([process.p,process.e])
