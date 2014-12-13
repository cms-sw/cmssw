import FWCore.ParameterSet.Config as cms
import DQM.TrackingMonitor.TrackingMonitor_cfi as _TrackMon
import DQMOffline.Alignment.muonAlignment_cfi as _MuonAl

from TrackingTools.MaterialEffects.Propagators_cff import *
from TrackingTools.GeomPropagators.SmartPropagator_cff import *
#from DQM.HLTEvF.HLTMonBitSummary_cfi import hltMonBitSummary

#Below all DQM modules for MuonAlignment AlCaRecos are instantiated.

##########################################################
#MuAlGlobalCosmicsInCollisions
##########################################################
ALCARECOMuAlGlobalCosmicsInCollisionsTrackingDQM = _TrackMon.TrackMon.clone(
    #names & designations    
    TrackProducer = 'ALCARECOMuAlGlobalCosmicsInCollisions:GlobalMuon',
    AlgoName = 'ALCARECOMuAlGlobalCosmicsInCollisions',
    FolderName = 'AlCaReco/MuAlGlobalCosmicsInCollisions',
    BSFolderName = "AlCaReco/MuAlGlobalCosmicsInCollisions/BeamSpot",
    MeasurementState = "default",
    doSeedParameterHistos = False,
    #sizes  
    TkSizeBin = 5,
    TkSizeMin = 0,
    TkSizeMax = 5,
    TrackPtBin = 100,
    TrackPtMin = 0,
    TrackPtMax = 30
    )

from Alignment.CommonAlignmentProducer.ALCARECOMuAlGlobalCosmicsInCollisions_cff import ALCARECOMuAlGlobalCosmicsInCollisionsHLT
#ALCARECOMuAlGlobalCosmicsInCollisionsHLTDQM = hltMonBitSummary.clone(
#    directory = "AlCaReco/MuAlGlobalCosmicsInCollisions/HLTSummary",
#    histLabel = "MuAlGlobalCosmicsInCollisions",
#    HLTPaths = ["HLT_.*L1.*"],
#    eventSetupPathsKey =  ALCARECOMuAlGlobalCosmicsInCollisionsHLT.eventSetupPathsKey.value()
#)
#ALCARECOMuAlGlobalCosmicsInCollisionsDQM = cms.Sequence( ALCARECOMuAlGlobalCosmicsInCollisionsTrackingDQM + ALCARECOMuAlGlobalCosmicsInCollisionsHLTDQM)
ALCARECOMuAlGlobalCosmicsInCollisionsDQM = cms.Sequence( ALCARECOMuAlGlobalCosmicsInCollisionsTrackingDQM)


##########################################################
#MuAlGlobalCosmics
##########################################################
ALCARECOMuAlGlobalCosmicsTrackingDQM = _TrackMon.TrackMon.clone(
    #names & designations    
    TrackProducer = 'ALCARECOMuAlGlobalCosmics:GlobalMuon',
    AlgoName = 'ALCARECOMuAlGlobalCosmics',
    FolderName = 'AlCaReco/MuAlGlobalCosmics',
    BSFolderName = "AlCaReco/MuAlGlobalCosmics/BeamSpot",
    MeasurementState = "default",
    doSeedParameterHistos = False,
    #sizes  
    TkSizeBin = 5,
    TkSizeMin = 0,
    TkSizeMax = 5,
    TrackPtBin = 100,
    TrackPtMin = 0,
    TrackPtMax = 30
    )

from Alignment.CommonAlignmentProducer.ALCARECOMuAlGlobalCosmics_cff import ALCARECOMuAlGlobalCosmicsHLT
#ALCARECOMuAlGlobalCosmicsHLTDQM = hltMonBitSummary.clone(
#    directory = "AlCaReco/MuAlGlobalCosmics/HLTSummary",
#    histLabel = "MuAlGlobalCosmics",
#    HLTPaths = ["HLT_.*L1.*"],
#    eventSetupPathsKey =  ALCARECOMuAlGlobalCosmicsHLT.eventSetupPathsKey.value()
#)
#ALCARECOMuAlGlobalCosmicsDQM = cms.Sequence( ALCARECOMuAlGlobalCosmicsTrackingDQM + ALCARECOMuAlGlobalCosmicsHLTDQM)
ALCARECOMuAlGlobalCosmicsDQM = cms.Sequence( ALCARECOMuAlGlobalCosmicsTrackingDQM)

##########################################################
################## MuAlBeamHalo ##########################
##########################################################
ALCARECOMuAlBeamHaloTrackingDQM = _TrackMon.TrackMon.clone(
    #names & designations  
    TrackProducer = 'ALCARECOMuAlBeamHalo',
    AlgoName = 'ALCARECOMuAlBeamHalo',
    FolderName = 'AlCaReco/MuAlBeamHalo',
    MeasurementState = "default",
    BSFolderName = "AlCaReco/MuAlBeamHalo/BeamSpot",
    doSeedParameterHistos = False,    
    #sizes
    TkSizeBin = 5,
    TkSizeMin = 0,
    TkSizeMax = 5,
    TrackPtBin = 100,
    TrackPtMin = 0,
    TrackPtMax = 30
    )

from Alignment.CommonAlignmentProducer.ALCARECOMuAlBeamHalo_cff import ALCARECOMuAlBeamHaloHLT
#ALCARECOMuAlBeamHaloHLTDQM = hltMonBitSummary.clone(
#    directory = "AlCaReco/MuAlBeamHalo/HLTSummary",
#    histLabel = "MuAlBeamHalo",
#    HLTPaths = ["HLT_.*Mu.*"],
#    eventSetupPathsKey =  ALCARECOMuAlBeamHaloHLT.eventSetupPathsKey.value()
#)
#ALCARECOMuAlBeamHaloDQM = cms.Sequence( ALCARECOMuAlBeamHaloTrackingDQM + ALCARECOMuAlBeamHaloHLTDQM)
ALCARECOMuAlBeamHaloDQM = cms.Sequence( ALCARECOMuAlBeamHaloTrackingDQM)


##########################################################
################# MuAlBeamHaloOverlaps ###################
##########################################################
ALCARECOMuAlBeamHaloOverlapsTrackingDQM = _TrackMon.TrackMon.clone(
    #names & designations  
    TrackProducer = 'ALCARECOMuAlBeamHaloOverlaps',
    AlgoName = 'ALCARECOMuAlBeamHaloOverlaps',
    FolderName= 'AlCaReco/MuAlBeamHaloOverlaps',
    MeasurementState = "default",
    BSFolderName = "AlCaReco/MuAlBeamHaloOverlaps/BeamSpot",
    doSeedParameterHistos = False,
    #sizes
    TkSizeBin = 70,
    TkSizeMin = 0,
    TkSizeMax = 70,
    TrackPtBin = 100,
    TrackPtMin = 0,
    TrackPtMax = 30
    )
from Alignment.CommonAlignmentProducer.ALCARECOMuAlBeamHaloOverlaps_cff import ALCARECOMuAlBeamHaloOverlapsHLT
#ALCARECOMuAlBeamHaloOverlapsHLTDQM = hltMonBitSummary.clone(
#    directory = "AlCaReco/MuAlBeamHaloOverlaps/HLTSummary",
#    histLabel = "MuAlBeamHaloOverlaps",
#    HLTPaths = ["HLT_.*L1.*"],
#    eventSetupPathsKey =  ALCARECOMuAlBeamHaloOverlapsHLT.eventSetupPathsKey.value()
#)
#ALCARECOMuAlBeamHaloOverlapsDQM = cms.Sequence( ALCARECOMuAlBeamHaloOverlapsTrackingDQM + ALCARECOMuAlBeamHaloOverlapsHLTDQM)
ALCARECOMuAlBeamHaloOverlapsDQM = cms.Sequence( ALCARECOMuAlBeamHaloOverlapsTrackingDQM)


##########################################################
################ MuAlCalIsolatedMu #######################
##########################################################
ALCARECOMuAlCalIsolatedMuTrackingDQM = _TrackMon.TrackMon.clone(
    #names & designations  
    TrackProducer = 'ALCARECOMuAlCalIsolatedMu:GlobalMuon',
    AlgoName = 'ALCARECOMuAlCalIsolatedMu',
    FolderName = 'AlCaReco/MuAlCalIsolatedMu',
    MeasurementState = "default",
    BSFolderName = "AlCaReco/MuAlCalIsolatedMu/BeamSpot",
    doSeedParameterHistos = False,
    #sizes			    
    TkSizeBin = 5,
    TkSizeMin = 0,
    TkSizeMax = 5,
    TrackPtBin = 100,
    TrackPtMin = 0,
    TrackPtMax = 100
    )

ALCARECOMuAlCalIsolatedMuMuAlDQM = _MuonAl.muonAlignment.clone(
    doSummary = True,
    FolderName ='AlCaReco/MuAlCalIsolatedMu',
    MuonCollection = 'ALCARECOMuAlCalIsolatedMu:GlobalMuon'
    )

from Alignment.CommonAlignmentProducer.ALCARECOMuAlCalIsolatedMu_cff import ALCARECOMuAlCalIsolatedMuHLT
#ALCARECOMuAlCalIsolatedMuHLTDQM = hltMonBitSummary.clone(
#    directory = "AlCaReco/MuAlCalIsolatedMu/HLTSummary",
#    histLabel = "MuAlCalIsolatedMu",
#    HLTPaths = ["HLT_.*Mu.*"],
#    eventSetupPathsKey =  ALCARECOMuAlCalIsolatedMuHLT.eventSetupPathsKey.value()
#)
#ALCARECOMuAlCalIsolatedMuDQM = cms.Sequence( ALCARECOMuAlCalIsolatedMuTrackingDQM + ALCARECOMuAlCalIsolatedMuMuAlDQM + ALCARECOMuAlCalIsolatedMuHLTDQM)
ALCARECOMuAlCalIsolatedMuDQM = cms.Sequence( ALCARECOMuAlCalIsolatedMuTrackingDQM + ALCARECOMuAlCalIsolatedMuMuAlDQM)


##########################################################
#################### MuAlOverlaps ########################
##########################################################
ALCARECOMuAlOverlapsTrackingDQM = _TrackMon.TrackMon.clone(
    #names & designations  
    TrackProducer = 'ALCARECOMuAlOverlaps',
    AlgoName = 'ALCARECOMuAlOverlaps',
    FolderName = 'AlCaReco/MuAlOverlaps',
    MeasurementState = "default",
    BSFolderName = "AlCaReco/MuAlOverlaps/BeamSpot",
    doSeedParameterHistos = False,
    #sizes                      
    TkSizeBin = 70,
    TkSizeMin = 0,
    TkSizeMax = 70,
    TrackPtBin = 100,
    TrackPtMin = 0,
    TrackPtMax = 30
    )

from Alignment.CommonAlignmentProducer.ALCARECOMuAlOverlaps_cff import ALCARECOMuAlOverlapsHLT
#ALCARECOMuAlOverlapsHLTDQM = hltMonBitSummary.clone(
#    directory = "AlCaReco/MuAlOverlaps/HLTSummary",
#    histLabel = "MuAlOverlaps",
#    HLTPaths = ["HLT_.*L1.*"],
#    eventSetupPathsKey =  ALCARECOMuAlOverlapsHLT.eventSetupPathsKey.value()
#)
#ALCARECOMuAlOverlapsDQM = cms.Sequence( ALCARECOMuAlOverlapsTrackingDQM + ALCARECOMuAlOverlapsHLTDQM)
ALCARECOMuAlOverlapsDQM = cms.Sequence( ALCARECOMuAlOverlapsTrackingDQM)


##########################################################
############### MuAlZMuMu ################################
##########################################################
ALCARECOMuAlZMuMuTrackingDQM= _TrackMon.TrackMon.clone(
    #names & designations  
    TrackProducer = 'ALCARECOMuAlZMuMu:StandAlone',
    AlgoName = 'ALCARECOMuAlZMuMu',
    FolderName = 'AlCaReco/MuAlZMuMu',
    MeasurementState = "default",
    BSFolderName = "AlCaReco/MuAlZMuMu/BeamSpot",
    doSeedParameterHistos = False,    
    #sizes                      
    TkSizeBin = 5,
    TkSizeMin = 0,
    TkSizeMax = 5,
    TrackPtBin = 100,
    TrackPtMin = 0,
    TrackPtMax = 100
    )

ALCARECOMuAlZMuMuMuAlDQM= _MuonAl.muonAlignment.clone(
    doSummary = True,
    FolderName ='AlCaReco/MuAlCaZMuMu',
    MuonCollection = 'ALCARECOMuAlZMuMu:StandAlone'
    )

from Alignment.CommonAlignmentProducer.ALCARECOMuAlZMuMu_cff import ALCARECOMuAlZMuMuHLT
#ALCARECOMuAlZMuMuHLTDQM = hltMonBitSummary.clone(
#    directory = "AlCaReco/MuAlZMuMu/HLTSummary",
#    histLabel = "MuAlZMuMu",
#    HLTPaths = ["HLT_.*Mu.*"],
#    eventSetupPathsKey =  ALCARECOMuAlZMuMuHLT.eventSetupPathsKey.value()
#)
#ALCARECOMuAlZMuMuDQM = cms.Sequence( ALCARECOMuAlZMuMuTrackingDQM + ALCARECOMuAlZMuMuMuAlDQM + ALCARECOMuAlZMuMuHLTDQM)
ALCARECOMuAlZMuMuDQM = cms.Sequence( ALCARECOMuAlZMuMuTrackingDQM + ALCARECOMuAlZMuMuMuAlDQM)

