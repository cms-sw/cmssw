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
process = cms.Process('ValidateL1Track')

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
     fileNames = cms.untracked.vstring('file:TenMuPt_0_100_ExtendedPhase2TkBE5D_10000_DIGI_L1_DIGI2RAW_L1TT_RECO.root')
#     fileNames = cms.untracked.vstring('file:TenMuPt_0_20_ExtendedPhase2TkBE5D_10000_DIGI_L1_DIGI2RAW_L1TT_RECO.root')
#     fileNames = cms.untracked.vstring('file:TenPiPt_0_50_ExtendedPhase2TkBE5D_5000_DIGI_L1_DIGI2RAW_L1TT_RECO.root')
#     fileNames = cms.untracked.vstring('file:TenElePt_0_50_ExtendedPhase2TkBE5D_5000_DIGI_L1_DIGI2RAW_L1TT_RECO.root')

#     fileNames = cms.untracked.vstring('file:TenMuPt_0_100_ExtendedPhase2TkBE5D_10000_DIGI_L1_DIGI2RAW_L1TT_RECO.root')

#     fileNames = cms.untracked.vstring('file:DYTauTau_ExtendedPhase2TkBE5D_750_DIGI_L1_DIGI2RAW_L1TT_RECO.root')


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
process.load('L1Trigger.TrackTrigger.TrackTrigger_cff')
process.load('SimTracker.TrackTriggerAssociation.TrackTriggerAssociator_cff')
process.L1TrackTrigger_step = cms.Path(process.TrackTriggerTracks)
process.L1TTAssociator_step = cms.Path(process.TrackTriggerAssociatorTracks)


#################################################################################################
# load the analyzer
#################################################################################################
process.ValidateL1Track = cms.EDAnalyzer("ValidateL1Track",
    DebugMode = cms.bool(True),
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
  fileName = cms.string('file:ValidateL1Track_ExtendedPhase2TkBE5D_Muon0100GeV3Wedges6SectorsNeighSeedCHECK.root')

#  fileName = cms.string('file:ValidateL1Track_ExtendedPhase2TkBE5D_Pion050GeV.root')
#  fileName = cms.string('file:ValidateL1Track_ExtendedPhase2TkBE5D_Ele.root')


#  fileName = cms.string('file:ValidateL1Track_ExtendedPhase2TkBE5D_Muon210100L1.root')


#  fileName = cms.string('file:ValidateL1Track_ExtendedPhase2TkBE5D_DYTauTau.root')

#fileName = cms.string('file:TEST.root')
)

#################################################################################################
# define the final path to be fed to cmsRun
#################################################################################################
process.p = cms.Path( process.ValidateL1Track )

process.schedule = cms.Schedule( process.L1TrackTrigger_step,process.L1TTAssociator_step,process.p )


