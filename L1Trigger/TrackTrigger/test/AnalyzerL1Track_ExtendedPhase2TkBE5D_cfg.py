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
#--------------------import os
process = cms.Process('AnalyzerL1Track')

#################################################################################################
# global tag
#################################################################################################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')

#################################################################################################
# load the specific tracker geometry
#################################################################################################
process.load('Configuration.Geometry.GeometryExtendedPhase2TkBE5DReco_cff')
process.load('Configuration.Geometry.GeometryExtendedPhase2TkBE5D_cff')
process.load('Geometry.TrackerGeometryBuilder.StackedTrackerGeometry_cfi')

#################################################################################################
# load the magnetic field
#################################################################################################
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
#process.load('Configuration.StandardSequences.MagneticField_40T_cff')
#-------------------process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

#################################################################################################
# define the source and maximum number of events to generate and simulate
#################################################################################################
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
     fileNames = cms.untracked.vstring('file:TenMuPt_0_20_ExtendedPhase2TkBE5D_10000_DIGI_L1_DIGI2RAW_L1TT_RECO.root')
#     fileNames = cms.untracked.vstring('file:TenMuPt_2_ExtendedPhase2TkBE5D_5000_DIGI_L1_DIGI2RAW_L1TT_RECO.root',
#                                       'file:TenMuPt_10_ExtendedPhase2TkBE5D_5000_DIGI_L1_DIGI2RAW_L1TT_RECO.root',
#                                       'file:TenMuPt_50_ExtendedPhase2TkBE5D_5000_DIGI_L1_DIGI2RAW_L1TT_RECO.root'
#     )
#     fileNames = cms.untracked.vstring( 'file:TenMuPt_0_50_ExtendedPhase2TkBE5D_5000_DIGI_L1_DIGI2RAW_L1TT_RECO.root',
#                                        'file:TenPiPt_0_50_ExtendedPhase2TkBE5D_5000_DIGI_L1_DIGI2RAW_L1TT_RECO.root',
#                                        'file:TenElePt_0_50_ExtendedPhase2TkBE5D_5000_DIGI_L1_DIGI2RAW_L1TT_RECO.root' )
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

################################################################################################
# produce the L1 Tracks
################################################################################################

process.BeamSpotFromSim = cms.EDProducer("BeamSpotFromSimProducer")
process.load('L1Trigger.TrackFindingTracklet.L1TTrack_cfi')
process.TrackFindingTracklet_step = cms.Path(process.BeamSpotFromSim*process.TTTracksFromPixelDigisTracklet)

process.load('L1Trigger.TrackFindingAM.L1AMTrack_cfi')
process.TrackFindingAM_step = cms.Path(process.TTTracksFromPixelDigisAM)

process.load('SimTracker.TrackTriggerAssociation.TrackTriggerAssociator_cff')
process.L1TTAssociator_step = cms.Path(process.TrackTriggerAssociatorTracks)


#################################################################################################
# load the analyzer
#################################################################################################
process.AnalyzerL1Track = cms.EDAnalyzer("AnalyzerL1Track",
    DebugMode = cms.bool(True),

    TTClusters       = cms.InputTag("TTStubsFromPixelDigis", "ClusterAccepted"),
    TTClusterMCTruth = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterAccepted"),
    TTStubs       = cms.InputTag("TTStubsFromPixelDigis", "StubAccepted"),
    TTStubMCTruth = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
    #TTTracks       = cms.InputTag("TTTracksFromPixelDigisTracklet", "TrackletBasedL1Tracks"),
    #TTTrackMCTruth = cms.InputTag("TTTrackAssociatorFromPixelDigis", "TrackletBasedL1Tracks"),

    TTTracks       = cms.InputTag("TTTracksFromPixelDigisAM", "AML1Tracks"),
    TTTrackMCTruth = cms.InputTag("TTTrackAssociatorFromPixelDigis", "AML1Tracks"),

    vLimitsPt  = cms.vdouble( 5.0,
                              15.0,
                              9999.99
    ),
    vStringPt  = cms.vstring( 'p_{T} in [0, 5) GeV/c',
                              'p_{T} in [5, 15) GeV/c',
                              'p_{T} in [15, ...) GeV/c'
    ),
    vLimitsEta = cms.vdouble( 0.8,
                              1.6,
                              9999.99
    ),
    vStringEta = cms.vstring( '|#eta| in [0, 0.8)',
                              '|#eta| in [0.8, 1.6)',
                              '|#eta| in [1.6, ...)'
    ),

    maxPtBinRes = cms.uint32( 20 ),
    ptBinSize   = cms.double( 2.5 ),

    maxEtaBinRes = cms.uint32( 30 ),
    etaBinSize   = cms.double ( 0.1 ),

)

#################################################################################################
# define output file and message logger
#################################################################################################
process.TFileService = cms.Service("TFileService",
  fileName = cms.string('file:AnalyzerL1Track_ExtendedPhase2TkBE5D_Muon0100GeVNEWFIT.root')
)

#################################################################################################
# define the final path to be fed to cmsRun
#################################################################################################
process.p = cms.Path( process.AnalyzerL1Track )

process.eca = cms.EDAnalyzer("EventContentAnalyzer")
process.eca_step = cms.Path(process.eca)

process.schedule = cms.Schedule(
process.TrackFindingTracklet_step,
process.TrackFindingAM_step,
process.L1TTAssociator_step,
#process.eca_step,
process.p
)

