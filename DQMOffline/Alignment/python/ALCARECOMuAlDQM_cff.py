import FWCore.ParameterSet.Config as cms
import DQM.TrackingMonitor.TrackingMonitor_cfi as _TrackMon
import DQMOffline.Alignment.muonAlignment_cfi as _MuonAl

from TrackingTools.MaterialEffects.Propagators_cff import *
from TrackingTools.GeomPropagators.SmartPropagator_cff import *

#Below all DQM modules for MuonAlignment AlCaRecos are instantiated.

##########################################################
############## MuAlStandAloneCosmics #####################
##########################################################
ALCARECOMuAlStandAloneCosmicsTrackingDQM = _TrackMon.TrackMon.clone(
    #names & designations  
    TrackProducer = 'ALCARECOMuAlStandAloneCosmics',
    AlgoName = 'ALCARECOMuAlStandAloneCosmics',
    FolderName = 'AlCaReco/MuAlStandAloneCosmics',
    #sizes
    TkSizeBin = 5,
    TkSizeMin = 0,
    TkSizeMax = 5,
    TrackPtBin = 100,
    TrackPtMin = 0,
    TrackPtMax = 100
    )

ALCARECOMuAlStandAloneCosmicsMuAlDQM = _MuonAl.muonAlignment.clone(
    doSummary = True,
    FolderName ='AlCaReco/MuAlStandAloneCosmics',
    MuonCollection = 'ALCARECOMuAlStandAloneCosmics'
 )

ALCARECOMuAlStandAloneCosmicsDQM = cms.Sequence(ALCARECOMuAlStandAloneCosmicsTrackingDQM  + ALCARECOMuAlStandAloneCosmicsMuAlDQM )

##########################################################
#MuAlGlobalCosmics
##########################################################
ALCARECOMuAlGlobalCosmicsDQM = _TrackMon.TrackMon.clone(
    #names & designations    
    TrackProducer = 'ALCARECOMuAlGlobalCosmics:GlobalMuon',
    AlgoName = 'ALCARECOMuAlGlobalCosmics',
    FolderName = 'AlCaReco/MuAlGlobalCosmics',
    #sizes  
    TkSizeBin = 5,
    TkSizeMin = 0,
    TkSizeMax = 5,
    TrackPtBin = 100,
    TrackPtMin = 0,
    TrackPtMax = 30
    )


##########################################################
################## MuAlBeamHalo ##########################
##########################################################
ALCARECOMuAlBeamHaloDQM = _TrackMon.TrackMon.clone(
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

##########################################################
################# MuAlBeamHaloOverlaps ###################
##########################################################
ALCARECOMuAlBeamHaloOverlapsDQM = _TrackMon.TrackMon.clone(
    #names & designations  
    TrackProducer = 'ALCARECOMuAlBeamHaloOverlaps',
    AlgoName = 'ALCARECOMuAlBeamHaloOverlaps',
    FolderName = 'AlCaReco/MuAlBeamHaloOverlaps',
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

##########################################################
################ MuAlCalIsolatedMu #######################
##########################################################
ALCARECOMuAlCalIsolatedMuTrackingDQM = _TrackMon.TrackMon.clone(
    #names & designations  
    TrackProducer = 'ALCARECOMuAlCalIsolatedMu:StandAlone',
    AlgoName = 'ALCARECOMuAlCalIsolatedMu',
    FolderName = 'AlCaReco/MuAlCalIsolatedMu',
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
    MuonCollection = 'ALCARECOMuAlCalIsolatedMu:StandAlone'
    )

ALCARECOMuAlCalIsolatedMuDQM = cms.Sequence( ALCARECOMuAlCalIsolatedMuTrackingDQM + ALCARECOMuAlCalIsolatedMuMuAlDQM )

##########################################################
#################### MuAlOverlaps ########################
##########################################################
ALCARECOMuAlOverlapsDQM = _TrackMon.TrackMon.clone(
    #names & designations  
    TrackProducer = 'ALCARECOMuAlOverlaps',
    AlgoName = 'ALCARECOMuAlOverlaps',
    FolderName = 'AlCaReco/MuAlOverlaps',
    #sizes                      
    TkSizeBin = 70,
    TkSizeMin = 0,
    TkSizeMax = 70,
    TrackPtBin = 100,
    TrackPtMin = 0,
    TrackPtMax = 30
    )

##########################################################
############### MuAlZMuMu ################################
##########################################################
ALCARECOMuAlZMuMuTrackingDQM= _TrackMon.TrackMon.clone(
    #names & designations  
    TrackProducer = 'ALCARECOMuAlZMuMu:StandAlone',
    AlgoName = 'ALCARECOMuAlZMuMu',
    FolderName = 'AlCaReco/MuAlZMuMu',
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

ALCARECOMuAlZMuMuDQM = cms.Sequence( ALCARECOMuAlZMuMuTrackingDQM + ALCARECOMuAlZMuMuMuAlDQM)
