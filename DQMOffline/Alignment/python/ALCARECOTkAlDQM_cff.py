import FWCore.ParameterSet.Config as cms
import DQM.TrackingMonitor.TrackingMonitor_cfi
import DQMOffline.Alignment.TkAlCaRecoMonitor_cfi
import DQMOffline.Alignment.DiMuonVertexMonitor_cfi
import DQMOffline.Alignment.DiMuonMassBiasMonitor_cfi

#Below all DQM modules for TrackerAlignment AlCaRecos are instantiated.
######################################################
#############---  TkAlZMuMu ---#######################
######################################################
__selectionName = 'TkAlZMuMu'
ALCARECOTkAlZMuMuTrackingDQM = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone(
#names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
    MeasurementState = "default",
    BSFolderName = "AlCaReco/"+__selectionName+"/BeamSpot",
    doSeedParameterHistos = False,
# Margins and settings
    TkSizeBin = 6,
    TkSizeMin = -0.5,
    TkSizeMax = 5.5,
    TrackPtBin = 150,
    TrackPtMin = 0,
    TrackPtMax = 150,
#choose histos from TrackingMonitor
    doAllPlots = True
)

ALCARECOTkAlZMuMuTkAlDQM =  DQMOffline.Alignment.TkAlCaRecoMonitor_cfi.TkAlCaRecoMonitor.clone(
#names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
# margins and settings
    runsOnReco = True,
    fillInvariantMass = True,
    MassBin = 300,
    MassMin = 50.0,
    MassMax = 150.0,
    SumChargeBin = 11,
    SumChargeMin = -5.5,
    SumChargeMax = 5.5,
    TrackPtBin= 150,
    TrackPtMin = 0.0,
    TrackPtMax = 150.0
)

ALCARECOTkAlZMuMuDQM = cms.Sequence( ALCARECOTkAlZMuMuTrackingDQM + ALCARECOTkAlZMuMuTkAlDQM )

#########################################################
#############--- TkAlDiMuonAndVertex ---#################
#########################################################
__selectionName = 'TkAlDiMuonAndVertex'
__trackCollName = 'TkAlDiMuon'

ALCARECOTkAlDiMuonAndVertexTkAlDQM =  DQMOffline.Alignment.TkAlCaRecoMonitor_cfi.TkAlCaRecoMonitor.clone(
#names and desigantions
    TrackProducer = 'ALCARECO'+__trackCollName,
    AlgoName = 'ALCARECO'+__trackCollName,
    FolderName = "AlCaReco/"+__selectionName,
# margins and settings
    runsOnReco = True,
    fillInvariantMass = True,
    MassBin = 300,
    MassMin = 50.0,
    MassMax = 150.0,
    SumChargeBin = 11,
    SumChargeMin = -5.5,
    SumChargeMax = 5.5,
    TrackPtBin= 150,
    TrackPtMin = 0.0,
    TrackPtMax = 150.0
)

ALCARECOTkAlDiMuonAndVertexVtxDQM = DQMOffline.Alignment.DiMuonVertexMonitor_cfi.DiMuonVertexMonitor.clone(
    muonTracks = 'ALCARECO'+__trackCollName,
    vertices = 'offlinePrimaryVertices',
    FolderName = "AlCaReco/"+__selectionName,
    maxSVdist = 50
)

ALCARECOTkAlDiMuonMassBiasDQM = DQMOffline.Alignment.DiMuonMassBiasMonitor_cfi.DiMuonMassBiasMonitor.clone(
    muonTracks = 'ALCARECO'+__trackCollName,
    FolderName = "AlCaReco/"+__selectionName
)

ALCARECOTkAlDiMuonAndVertexDQM = cms.Sequence(ALCARECOTkAlDiMuonAndVertexTkAlDQM + ALCARECOTkAlDiMuonAndVertexVtxDQM + ALCARECOTkAlDiMuonMassBiasDQM)

#########################################################
#############---  TkAlZMuMuHI ---########################
#########################################################
__selectionName = 'TkAlZMuMuHI'
ALCARECOTkAlZMuMuHITrackingDQM = ALCARECOTkAlZMuMuTrackingDQM.clone(
#names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
    BSFolderName = "AlCaReco/"+__selectionName+"/BeamSpot",
    allTrackProducer = "hiGeneralTracks" ,
    primaryVertex = 'hiSelectedVertex'
)

ALCARECOTkAlZMuMuHITkAlDQM = ALCARECOTkAlZMuMuTkAlDQM.clone(
#names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
    ReferenceTrackProducer= "hiGeneralTracks",
    CaloJetCollection= "iterativeConePu5CaloJets"
)

ALCARECOTkAlZMuMuHIDQM = cms.Sequence( ALCARECOTkAlZMuMuHITrackingDQM + ALCARECOTkAlZMuMuHITkAlDQM )

#########################################################
#############---  TkAlZMuMuPA ---########################
#########################################################
__selectionName = 'TkAlZMuMuPA'
ALCARECOTkAlZMuMuPATrackingDQM = ALCARECOTkAlZMuMuTrackingDQM.clone(
    #names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
    BSFolderName = "AlCaReco/"+__selectionName+"/BeamSpot"
)

ALCARECOTkAlZMuMuPATkAlDQM = ALCARECOTkAlZMuMuTkAlDQM.clone(
    #names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName
)

ALCARECOTkAlZMuMuPADQM = cms.Sequence( ALCARECOTkAlZMuMuPATrackingDQM + ALCARECOTkAlZMuMuPATkAlDQM )

#########################################################
#############---  TkAlJpsiMuMu ---#######################
#########################################################
__selectionName = 'TkAlJpsiMuMu'
ALCARECOTkAlJpsiMuMuTrackingDQM = ALCARECOTkAlZMuMuTrackingDQM.clone(
    #names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
    BSFolderName = "AlCaReco/"+__selectionName+"/BeamSpot",
    # margins and settings
    TrackPtMax = 50
)
ALCARECOTkAlJpsiMuMuTkAlDQM = ALCARECOTkAlZMuMuTkAlDQM.clone(
    #names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
    # margins and settings
    MassMin = 2.5,
    MassMax = 4.0,
    TrackPtMax = 50
)

ALCARECOTkAlJpsiMuMuVtxDQM = DQMOffline.Alignment.DiMuonVertexMonitor_cfi.DiMuonVertexMonitor.clone(
    muonTracks = 'ALCARECO'+__selectionName,
    decayMotherName = "J/#psi",
    vertices = 'offlinePrimaryVertices',
    FolderName = "AlCaReco/"+__selectionName,
    maxSVdist = 50
)

ALCARECOTkAlJpsiMassBiasDQM = DQMOffline.Alignment.DiMuonMassBiasMonitor_cfi.DiMuonMassBiasMonitor.clone(
    muonTracks = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
    decayMotherName = 'J/#psi',
    DiMuMassConfig = dict(ymin = 2.7 ,ymax = 3.4, maxDeltaEta = 1.3))

ALCARECOTkAlJpsiMuMuDQM = cms.Sequence( ALCARECOTkAlJpsiMuMuTrackingDQM + ALCARECOTkAlJpsiMuMuTkAlDQM + ALCARECOTkAlJpsiMuMuVtxDQM + ALCARECOTkAlJpsiMassBiasDQM)

#########################################################
#############---  TkAlJpsiMuMuHI ---#####################
#########################################################
__selectionName = 'TkAlJpsiMuMuHI'
ALCARECOTkAlJpsiMuMuHITrackingDQM = ALCARECOTkAlZMuMuTrackingDQM.clone(
    #names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
    BSFolderName = "AlCaReco/"+__selectionName+"/BeamSpot",
    allTrackProducer = "hiGeneralTracks",
    primaryVertex = 'hiSelectedVertex',
    # margins and settings
    TrackPtMax = 50
)

ALCARECOTkAlJpsiMuMuHITkAlDQM = ALCARECOTkAlZMuMuTkAlDQM.clone(
#names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
    ReferenceTrackProducer= "hiGeneralTracks",
    CaloJetCollection= "iterativeConePu5CaloJets",
# margins and settings
    MassMin = 2.5,
    MassMax = 4.0,
    TrackPtMax = 50
)

ALCARECOTkAlJpsiMuMuHIDQM = cms.Sequence( ALCARECOTkAlJpsiMuMuHITrackingDQM + ALCARECOTkAlJpsiMuMuHITkAlDQM )

############################################################
#############---  TkAlUpsilonMuMu ---#######################
############################################################
__selectionName = 'TkAlUpsilonMuMu'
ALCARECOTkAlUpsilonMuMuTrackingDQM = ALCARECOTkAlJpsiMuMuTrackingDQM.clone(
#names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
    BSFolderName = "AlCaReco/"+__selectionName+"/BeamSpot"
)

ALCARECOTkAlUpsilonMuMuTkAlDQM = ALCARECOTkAlZMuMuTkAlDQM.clone(
#names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
# margins and settings
    MassMin = 8.,
    MassMax = 10,
    TrackPtMax = 50
)

ALCARECOTkAlUpsilonMuMuVtxDQM = DQMOffline.Alignment.DiMuonVertexMonitor_cfi.DiMuonVertexMonitor.clone(
    muonTracks = 'ALCARECO'+__selectionName,
    decayMotherName = "#Upsilon",
    vertices = 'offlinePrimaryVertices',
    FolderName = "AlCaReco/"+__selectionName,
    maxSVdist = 50
)

ALCARECOTkAlUpsilonMassBiasDQM = DQMOffline.Alignment.DiMuonMassBiasMonitor_cfi.DiMuonMassBiasMonitor.clone(
    muonTracks = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
    decayMotherName = '#Upsilon',
    DiMuMassConfig = dict(ymin = 8.9 ,ymax = 9.9, maxDeltaEta = 1.6))

ALCARECOTkAlUpsilonMuMuDQM = cms.Sequence( ALCARECOTkAlUpsilonMuMuTrackingDQM + ALCARECOTkAlUpsilonMuMuTkAlDQM + ALCARECOTkAlUpsilonMuMuVtxDQM + ALCARECOTkAlUpsilonMassBiasDQM)

############################################################
#############---  TkAlUpsilonMuMuHI ---#####################
############################################################
__selectionName = 'TkAlUpsilonMuMuHI'
ALCARECOTkAlUpsilonMuMuHITrackingDQM = ALCARECOTkAlJpsiMuMuHITrackingDQM.clone(
    #names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
    BSFolderName = "AlCaReco/"+__selectionName+"/BeamSpot",
    allTrackProducer = "hiGeneralTracks",
    primaryVertex = 'hiSelectedVertex',
    # margins and settings
    TrackPtMax = 50
)

ALCARECOTkAlUpsilonMuMuHITkAlDQM = ALCARECOTkAlZMuMuTkAlDQM.clone(
    #names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
    ReferenceTrackProducer= "hiGeneralTracks",
    CaloJetCollection= "iterativeConePu5CaloJets",
    # margins and settings
    MassMin = 8.,
    MassMax = 10,
    TrackPtMax = 50
)

ALCARECOTkAlUpsilonMuMuHIDQM = cms.Sequence( ALCARECOTkAlUpsilonMuMuHITrackingDQM + ALCARECOTkAlUpsilonMuMuHITkAlDQM )

############################################################
#############---  TkAlUpsilonMuMuPA ---#####################
############################################################
__selectionName = 'TkAlUpsilonMuMuPA'
ALCARECOTkAlUpsilonMuMuPATrackingDQM = ALCARECOTkAlUpsilonMuMuTrackingDQM.clone(
    #names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
    BSFolderName = "AlCaReco/"+__selectionName+"/BeamSpot",
    # margins and settings
    TrackPtMax = 50
)

ALCARECOTkAlUpsilonMuMuPATkAlDQM = ALCARECOTkAlZMuMuTkAlDQM.clone(
    #names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
    # margins and settings
    MassMin = 8.,
    MassMax = 10,
    TrackPtMax = 50
)

ALCARECOTkAlUpsilonMuMuPADQM = cms.Sequence( ALCARECOTkAlUpsilonMuMuPATrackingDQM + ALCARECOTkAlUpsilonMuMuPATkAlDQM )

#########################################################
#############---  TkAlBeamHalo ---#######################
#########################################################
__selectionName = 'TkAlBeamHalo'
ALCARECOTkAlBeamHaloTrackingDQM = ALCARECOTkAlZMuMuTrackingDQM.clone(
    #names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
    BSFolderName = "AlCaReco/"+__selectionName+"/BeamSpot"
)


ALCARECOTkAlBeamHaloDQM = cms.Sequence( ALCARECOTkAlBeamHaloTrackingDQM )

########################################################
#############---  TkAlMinBias ---#######################
########################################################
__selectionName = 'TkAlMinBias'
ALCARECOTkAlMinBiasTrackingDQM = ALCARECOTkAlZMuMuTrackingDQM.clone(
    #names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
    BSFolderName = "AlCaReco/"+__selectionName+"/BeamSpot",
    # margins and settings
    TkSizeBin = 71,
    TkSizeMin = -0.5,
    TkSizeMax = 70.5,
    TrackPtMax = 30
)

ALCARECOTkAlMinBiasTkAlDQM = ALCARECOTkAlZMuMuTkAlDQM.clone(
    #names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
    # margins and settings
    fillInvariantMass = False,
    TrackPtMax = 30,
    SumChargeBin = 101,
    SumChargeMin = -50.5,
    SumChargeMax = 50.5
)

ALCARECOTkAlMinBiasDQM = cms.Sequence( ALCARECOTkAlMinBiasTrackingDQM + ALCARECOTkAlMinBiasTkAlDQM )

########################################################
#############---  TkAlJetHT ---#######################
########################################################
__selectionName = 'TkAlJetHT'
ALCARECOTkAlJetHTTrackingDQM = ALCARECOTkAlZMuMuTrackingDQM.clone(
    #names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
    BSFolderName = "AlCaReco/"+__selectionName+"/BeamSpot",
    # margins and settings
    TkSizeBin = 71,
    TkSizeMin = -0.5,
    TkSizeMax = 70.5,
    TrackPtMax = 30
)

ALCARECOTkAlJetHTTkAlDQM = ALCARECOTkAlZMuMuTkAlDQM.clone(
    #names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
    # margins and settings
    fillInvariantMass = False,
    TrackPtMax = 30,
    SumChargeBin = 101,
    SumChargeMin = -50.5,
    SumChargeMax = 50.5
)

ALCARECOTkAlJetHTDQM = cms.Sequence( ALCARECOTkAlJetHTTrackingDQM + ALCARECOTkAlJetHTTkAlDQM)

########################################################
#############---  TkAlMinBiasHI ---#####################
########################################################
__selectionName = 'TkAlMinBiasHI'
ALCARECOTkAlMinBiasHITrackingDQM = ALCARECOTkAlMinBiasTrackingDQM.clone(
    #names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
    BSFolderName = "AlCaReco/"+__selectionName+"/BeamSpot",
    primaryVertex = "hiSelectedVertex",
    allTrackProducer = "hiGeneralTracks",
    # margins and settings
    TkSizeBin = 71,
    TkSizeMin = -0.5,
    TkSizeMax = 70.5,
    TrackPtMax = 30
)

ALCARECOTkAlMinBiasHITkAlDQM = ALCARECOTkAlMinBiasTkAlDQM.clone(
    #names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    ReferenceTrackProducer = 'hiGeneralTracks',
    CaloJetCollection = 'iterativeConePu5CaloJets',
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
    # margins and settings
    fillInvariantMass = False,
    TrackPtMax = 30,
    SumChargeBin = 101,
    SumChargeMin = -50.5,
    SumChargeMax = 50.5
)

ALCARECOTkAlMinBiasHIDQM = cms.Sequence( ALCARECOTkAlMinBiasHITrackingDQM + ALCARECOTkAlMinBiasHITkAlDQM )

#############################################################
#############---  TkAlMuonIsolated ---#######################
#############################################################
__selectionName = 'TkAlMuonIsolated'
ALCARECOTkAlMuonIsolatedTrackingDQM = ALCARECOTkAlZMuMuTrackingDQM.clone(
    #names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
    BSFolderName = "AlCaReco/"+__selectionName+"/BeamSpot",
    # margins and settings
    TkSizeBin = 16,
    TkSizeMin = -0.5,
    TkSizeMax = 15.5
)
ALCARECOTkAlMuonIsolatedTkAlDQM = ALCARECOTkAlMinBiasTkAlDQM.clone(
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName
)

ALCARECOTkAlMuonIsolatedDQM = cms.Sequence( ALCARECOTkAlMuonIsolatedTrackingDQM + ALCARECOTkAlMuonIsolatedTkAlDQM )

#############################################################
#############---  TkAlMuonIsolatedHI ---#####################
#############################################################
__selectionName = 'TkAlMuonIsolatedHI'
ALCARECOTkAlMuonIsolatedHITrackingDQM = ALCARECOTkAlZMuMuTrackingDQM.clone(
    #names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
    BSFolderName = "AlCaReco/"+__selectionName+"/BeamSpot",
    allTrackProducer = "hiGeneralTracks",
    primaryVertex = 'hiSelectedVertex',
    # margins and settings
    TkSizeBin = 16,
    TkSizeMin = -0.5,
    TkSizeMax = 15.5
)
ALCARECOTkAlMuonIsolatedHITkAlDQM = ALCARECOTkAlMinBiasTkAlDQM.clone(
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
    ReferenceTrackProducer= "hiGeneralTracks",
    CaloJetCollection= "iterativeConePu5CaloJets" 
)

ALCARECOTkAlMuonIsolatedHIDQM = cms.Sequence( ALCARECOTkAlMuonIsolatedHITrackingDQM + ALCARECOTkAlMuonIsolatedHITkAlDQM )

#############################################################
#############---  TkAlMuonIsolatedPA ---#####################
#############################################################
__selectionName = 'TkAlMuonIsolatedPA'
ALCARECOTkAlMuonIsolatedPATrackingDQM = ALCARECOTkAlZMuMuTrackingDQM.clone(
    #names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
    BSFolderName = "AlCaReco/"+__selectionName+"/BeamSpot",
    # margins and settings
    TkSizeBin = 16,
    TkSizeMin = -0.5,
    TkSizeMax = 15.5
)
ALCARECOTkAlMuonIsolatedPATkAlDQM = ALCARECOTkAlMinBiasTkAlDQM.clone(
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName
)

ALCARECOTkAlMuonIsolatedPADQM = cms.Sequence( ALCARECOTkAlMuonIsolatedPATrackingDQM + ALCARECOTkAlMuonIsolatedPATkAlDQM )

####################################################
#############---  TkAlLAS ---#######################
####################################################
import DQMOffline.Alignment.LaserAlignmentT0ProducerDQM_cfi
__selectionName = 'TkAlLAS'
ALCARECOTkAlLASDigiDQM= DQMOffline.Alignment.LaserAlignmentT0ProducerDQM_cfi.LaserAlignmentT0ProducerDQM.clone(
    # names and designation
    FolderName = "AlCaReco/"+__selectionName,
    # settings
    LowerAdcThreshold = 15,
    UpperAdcThreshold = 220,
    DigiProducerList = cms.VPSet(
        cms.PSet(
            DigiLabel = cms.string( 'ZeroSuppressed' ),
            DigiType = cms.string( 'Processed' ),
            DigiProducer = cms.string( 'ALCARECOTkAlLAST0Producer' )
        )
    )
)
ALCARECOTkAlLASDQM = cms.Sequence( ALCARECOTkAlLASDigiDQM )

##################################################################
###### DQM modules for cosmic data taking during collisions ######
##################################################################
###############################
### TkAlCosmicsInCollisions ###
###############################
__selectionName = 'TkAlCosmicsInCollisions'
ALCARECOTkAlCosmicsInCollisionsTrackingDQM = ALCARECOTkAlZMuMuTrackingDQM.clone(
    #names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = 'AlCaReco/TkAlCosmicsInCollisions',
    BSFolderName = "AlCaReco/"+__selectionName+"/BeamSpot",
    # margins and settings
    TrackPtBin = 500,
    TrackPtMin = 0,
    TrackPtMax = 500
)
ALCARECOTkAlCosmicsInCollisionsTkAlDQM = ALCARECOTkAlMinBiasTkAlDQM.clone(
#names and desigantions
    FolderName = 'AlCaReco/TkAlCosmicsInCollisions',
    TrackProducer = 'ALCARECO'+__selectionName,
    ReferenceTrackProducer = 'cosmicDCTracks',
    AlgoName = 'ALCARECO'+__selectionName
)

ALCARECOTkAlCosmicsInCollisionsDQM = cms.Sequence( ALCARECOTkAlCosmicsInCollisionsTrackingDQM + ALCARECOTkAlCosmicsInCollisionsTkAlDQM )

################################################
###### DQM modules for cosmic data taking ######
################################################
########################
### TkAlCosmicsCTF0T ###
########################
__selectionName = 'TkAlCosmicsCTF0T'
ALCARECOTkAlCosmicsCTF0TTrackingDQM = ALCARECOTkAlZMuMuTrackingDQM.clone(
    #names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = 'AlCaReco/TkAlCosmics0T',
    BSFolderName = "AlCaReco/"+__selectionName+"/BeamSpot",
    # margins and settings
    TrackPtBin = 500,
    TrackPtMin = 0,
    TrackPtMax = 500
)
ALCARECOTkAlCosmicsCTF0TTkAlDQM = ALCARECOTkAlMinBiasTkAlDQM.clone(
    #names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    ReferenceTrackProducer = 'ctfWithMaterialTracksP5',
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = 'AlCaReco/TkAlCosmics0T',
    # margins and settings
    useSignedR = True
)

ALCARECOTkAlCosmicsCTF0TDQM = cms.Sequence( ALCARECOTkAlCosmicsCTF0TTrackingDQM + ALCARECOTkAlCosmicsCTF0TTkAlDQM )

#############################
### TkAlCosmicsCosmicTF0T ###
#############################
__selectionName = 'TkAlCosmicsCosmicTF0T'
ALCARECOTkAlCosmicsCosmicTF0TTrackingDQM = ALCARECOTkAlCosmicsCTF0TTrackingDQM.clone(
#names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    BSFolderName = "AlCaReco/"+__selectionName+"/BeamSpot",
)
ALCARECOTkAlCosmicsCosmicTF0TTkAlDQM = ALCARECOTkAlCosmicsCTF0TTkAlDQM.clone(
#names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    ReferenceTrackProducer = 'cosmictrackfinderP5',
    AlgoName = 'ALCARECO'+__selectionName
)

ALCARECOTkAlCosmicsCosmicTF0TDQM = cms.Sequence( ALCARECOTkAlCosmicsCosmicTF0TTrackingDQM + ALCARECOTkAlCosmicsCosmicTF0TTkAlDQM )

#############################
### TkAlCosmicsRegional0T ###
#############################
__selectionName = 'TkAlCosmicsRegional0T'
ALCARECOTkAlCosmicsRegional0TTrackingDQM = ALCARECOTkAlCosmicsCTF0TTrackingDQM.clone(
#names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    BSFolderName = "AlCaReco/"+__selectionName+"/BeamSpot",
)
ALCARECOTkAlCosmicsRegional0TTkAlDQM = ALCARECOTkAlCosmicsCTF0TTkAlDQM.clone(
#names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    ReferenceTrackProducer = 'cosmictrackfinderP5',
    AlgoName = 'ALCARECO'+__selectionName
)

ALCARECOTkAlCosmicsRegional0TDQM = cms.Sequence( ALCARECOTkAlCosmicsRegional0TTrackingDQM + ALCARECOTkAlCosmicsRegional0TTkAlDQM )

#####################################
### TkAlCosmicsDuringCollisions0T ###
#####################################
__selectionName = 'TkAlCosmicsDuringCollisions0T'
ALCARECOTkAlCosmicsDuringCollisions0TTrackingDQM = ALCARECOTkAlCosmicsCTF0TTrackingDQM.clone(
#names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    BSFolderName = "AlCaReco/"+__selectionName+"/BeamSpot",
)
ALCARECOTkAlCosmicsDuringCollisions0TTkAlDQM = ALCARECOTkAlCosmicsCTF0TTkAlDQM.clone(
#names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    ReferenceTrackProducer = 'cosmictrackfinderP5',
    AlgoName = 'ALCARECO'+__selectionName
)

ALCARECOTkAlCosmicsDuringCollisions0TDQM = cms.Sequence( ALCARECOTkAlCosmicsDuringCollisions0TTrackingDQM + ALCARECOTkAlCosmicsDuringCollisions0TTkAlDQM )

##########################################################################
###### DQM modules for cosmic data taking with momentum measurement ######
##########################################################################
######################
### TkAlCosmicsCTF ###
######################
__selectionName = 'TkAlCosmicsCTF'
ALCARECOTkAlCosmicsCTFTrackingDQM = ALCARECOTkAlCosmicsCTF0TTrackingDQM.clone(
    #names and desigantions
    FolderName = 'AlCaReco/TkAlCosmics',
    BSFolderName = "AlCaReco/"+__selectionName+"/BeamSpot",
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName
)
ALCARECOTkAlCosmicsCTFTkAlDQM = ALCARECOTkAlCosmicsCTF0TTkAlDQM.clone(
    #names and desigantions
    FolderName = 'AlCaReco/TkAlCosmics',
    TrackProducer = 'ALCARECO'+__selectionName,
    ReferenceTrackProducer = ALCARECOTkAlCosmicsCTF0TTkAlDQM.ReferenceTrackProducer,
    AlgoName = 'ALCARECO'+__selectionName
)

ALCARECOTkAlCosmicsCTFDQM = cms.Sequence( ALCARECOTkAlCosmicsCTFTrackingDQM + ALCARECOTkAlCosmicsCTFTkAlDQM )

###########################
### TkAlCosmicsCosmicTF ###
###########################
__selectionName = 'TkAlCosmicsCosmicTF'
ALCARECOTkAlCosmicsCosmicTFTrackingDQM = ALCARECOTkAlCosmicsCTFTrackingDQM.clone(
    #names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    BSFolderName = "AlCaReco/"+__selectionName+"/BeamSpot"
)
ALCARECOTkAlCosmicsCosmicTFTkAlDQM = ALCARECOTkAlCosmicsCosmicTF0TTkAlDQM.clone(
    #names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    ReferenceTrackProducer = ALCARECOTkAlCosmicsCosmicTF0TTkAlDQM.ReferenceTrackProducer,
    AlgoName = 'ALCARECO'+__selectionName
)

ALCARECOTkAlCosmicsCosmicTFDQM = cms.Sequence( ALCARECOTkAlCosmicsCosmicTFTrackingDQM + ALCARECOTkAlCosmicsCosmicTFTkAlDQM )

###########################
### TkAlCosmicsRegional ###
###########################
__selectionName = 'TkAlCosmicsRegional'
ALCARECOTkAlCosmicsRegionalTrackingDQM = ALCARECOTkAlCosmicsCTFTrackingDQM.clone(
    #names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    BSFolderName = "AlCaReco/"+__selectionName+"/BeamSpot"
)
ALCARECOTkAlCosmicsRegionalTkAlDQM = ALCARECOTkAlCosmicsRegional0TTkAlDQM.clone(
    #names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    ReferenceTrackProducer = ALCARECOTkAlCosmicsRegional0TTkAlDQM.ReferenceTrackProducer,
    AlgoName = 'ALCARECO'+__selectionName
)

ALCARECOTkAlCosmicsRegionalDQM = cms.Sequence( ALCARECOTkAlCosmicsRegionalTrackingDQM + ALCARECOTkAlCosmicsRegionalTkAlDQM )

