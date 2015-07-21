muonPtString = '20'

####################################################
# SLHCUpgradeSimulations                           #
# Configuration file for Full Workflow             #
# Step 2 (again)                                   #
# Understand if everything is fine with            #
# L1TkCluster e L1TkStub                           #
####################################################
# Nicola Pozzobon                                  #
# CERN, August 2012                                #
####################################################

#################################################################################################
# import of general framework
#################################################################################################
import FWCore.ParameterSet.Config as cms

#################################################################################################
# name the process
#################################################################################################
process = cms.Process('AnalyzerDTMatches')

#################################################################################################
# global tag
#################################################################################################
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')

#################################################################################################
# load the specific tracker geometry
#################################################################################################
process.load('Configuration.Geometry.GeometryExtended2023TTIReco_cff')
process.load('Configuration.Geometry.GeometryExtended2023TTI_cff')
process.load('Geometry.TrackerGeometryBuilder.StackedTrackerGeometry_cfi')

#################################################################################################
# load the magnetic field
#################################################################################################
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
#process.load('Configuration.StandardSequences.MagneticField_40T_cff')
#------------------process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

#################################################################################################
# define the source and maximum number of events to process
#################################################################################################
process.source = cms.Source(
    "PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring(
#'/store/user/pozzo/MUPGUN_14_Extended2023TTI_6_2_0_SLHC12_GEN_SIM_V002/MUPGUN_14_Extended2023TTI_6_2_0_SLHC12_DIGI_L1_DIGI2RAW_L1TT_RECO_V001G/95a3b5071be09d95bfad13b012bdbdd3/TwoMuPt_14_Extended2023TTI_500000_DIGI_L1_DIGI2RAW_L1TT_RECO_V001G_88_1_VtZ.root',
'/store/user/pozzo/MUPGUN_14_Extended2023TTI_6_2_0_SLHC12_GEN_SIM_V002/MUPGUN_14_Extended2023TTI_6_2_0_SLHC12_DIGI_L1_DIGI2RAW_L1TT_RECO_V001G/95a3b5071be09d95bfad13b012bdbdd3/TwoMuPt_14_Extended2023TTI_500000_DIGI_L1_DIGI2RAW_L1TT_RECO_V001G_89_1_FG5.root',
       )
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-2)#2)
)

#################################################################################################
# load the analyzer
#################################################################################################
process.AnalyzerDTMatches = cms.EDAnalyzer("AnalyzerDTMatches",
    BothCharges = cms.bool(False),
    GetPositive = cms.bool(True),
)

#################################################################################################
# define output file and message logger
#################################################################################################
process.TFileService = cms.Service("TFileService",
  fileName = cms.string('file:AnalyzerDTMatches_Extended2023TTI_'+muonPtString+'.root')
)

#################################################################################################
# now, all the DT related stuff
#################################################################################################
# to produce, in case, collection of L1MuDTTrack objects:
#process.dttfDigis = cms.Path(process.simDttfDigis)

# the DT geometry
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("SimMuon/DTDigitizer/muonDTDigis_cfi")
##process.load("L1TriggerConfig.DTTPGConfigProducers.L1DTTPGConfig_cff")

#################################################################################################
# define the producer of DT + TK objects
#################################################################################################
process.load("L1Trigger.DTPlusTrackTrigger.DTPlusTrackProducer_cfi")
#process.DTPlusTrackProducer.useRoughTheta = cms.untracked.bool(True)
process.DTPlusTk_step = cms.Path(process.DTPlusTrackProducer)

#################################################################################################
# produce the L1 Tracks
#################################################################################################
process.load('Configuration.StandardSequences.L1TrackTrigger_cff')
process.TrackFindingTracklet_step = cms.Path(process.TrackTriggerTTTracks)

#process.load('L1Trigger.TrackFindingAM.L1AMTrack_cfi')
#process.TrackFindingAM_step = cms.Path(process.TTTracksFromPixelDigisAM)

process.load('SimTracker.TrackTriggerAssociation.TrackTriggerAssociator_cff')
process.TTTrackAssociatorFromPixelDigis = cms.EDProducer("TTTrackAssociator_PixelDigi_",
    TTTracks = cms.VInputTag(cms.InputTag("TTTracksFromPixelDigis","Level1TTTracks")),
    TTStubTruth = cms.InputTag("TTStubAssociatorFromPixelDigis","StubAccepted"),
    TTClusterTruth = cms.InputTag("TTClusterAssociatorFromPixelDigis","ClusterAccepted")
)
process.L1TTAssociator_step = cms.Path(process.TrackTriggerAssociatorTracks)

#################################################################################################
# Make the job crash in case of missing product
#################################################################################################
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound')
)

#################################################################################################
# define the final path to be fed to cmsRun
#################################################################################################
process.Plots_step = cms.Path( process.AnalyzerDTMatches )


process.eca = cms.EDAnalyzer("EventContentAnalyzer")
process.eca_step = cms.Path(process.eca)

#################################################################################################
process.schedule = cms.Schedule(
# process.TrackFindingTracklet_step,
# process.TrackFindingAM_step,
# process.L1TTAssociator_step,
 process.DTPlusTk_step,
 process.Plots_step,
# process.eca_step,
#process.end
)

