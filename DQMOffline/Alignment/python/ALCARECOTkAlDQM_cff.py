import FWCore.ParameterSet.Config as cms
import DQM.TrackingMonitor.TrackingMonitor_cfi
import DQMOffline.Alignment.TkAlCaRecoMonitor_cfi

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
#############---  TkAlJpsiMuMu ---#######################
#########################################################
__selectionName = 'TkAlJpsiMuMu'
ALCARECOTkAlJpsiMuMuTrackingDQM = ALCARECOTkAlZMuMuTrackingDQM.clone(
#names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
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
ALCARECOTkAlJpsiMuMuDQM = cms.Sequence( ALCARECOTkAlJpsiMuMuTrackingDQM + ALCARECOTkAlJpsiMuMuTkAlDQM )

############################################################
#############---  TkAlUpsilonMuMu ---#######################
############################################################
__selectionName = 'TkAlUpsilonMuMu'
ALCARECOTkAlUpsilonMuMuTrackingDQM = ALCARECOTkAlJpsiMuMuTrackingDQM.clone(
#names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName
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
ALCARECOTkAlUpsilonMuMuDQM = cms.Sequence( ALCARECOTkAlUpsilonMuMuTrackingDQM + ALCARECOTkAlUpsilonMuMuTkAlDQM)


#########################################################
#############---  TkAlBeamHalo ---#######################
#########################################################
__selectionName = 'TkAlBeamHalo'
ALCARECOTkAlBeamHaloTrackingDQM = ALCARECOTkAlZMuMuTrackingDQM.clone(
#names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName
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
ALCARECOTkAlMinBiasDQM = cms.Sequence( ALCARECOTkAlMinBiasTrackingDQM + ALCARECOTkAlMinBiasTkAlDQM)

#############################################################
#############---  TkAlMuonIsolated ---#######################
#############################################################
__selectionName = 'TkAlMuonIsolated'
ALCARECOTkAlMuonIsolatedTrackingDQM = ALCARECOTkAlZMuMuTrackingDQM.clone(
#names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
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
ALCARECOTkAlMuonIsolatedDQM = cms.Sequence( ALCARECOTkAlMuonIsolatedTrackingDQM + ALCARECOTkAlMuonIsolatedTkAlDQM)

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
    cms.PSet(DigiLabel = cms.string( 'ZeroSuppressed' ),
             DigiType = cms.string( 'Processed' ),
             DigiProducer = cms.string( 'laserAlignmentT0Producer' )
             )
    )
)
ALCARECOTkAlLASDQM = cms.Sequence( ALCARECOTkAlLASDigiDQM )

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
ALCARECOTkAlCosmicsCTF0TDQM = cms.Sequence( ALCARECOTkAlCosmicsCTF0TTrackingDQM + ALCARECOTkAlCosmicsCTF0TTkAlDQM)

#############################
### TkAlCosmicsCosmicTF0T ###
#############################
__selectionName = 'TkAlCosmicsCosmicTF0T'
ALCARECOTkAlCosmicsCosmicTF0TTrackingDQM = ALCARECOTkAlCosmicsCTF0TTrackingDQM.clone(
#names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName
)
ALCARECOTkAlCosmicsCosmicTF0TTkAlDQM = ALCARECOTkAlCosmicsCTF0TTkAlDQM.clone(
#names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    ReferenceTrackProducer = 'cosmictrackfinderP5',
    AlgoName = 'ALCARECO'+__selectionName
)
ALCARECOTkAlCosmicsCosmicTF0TDQM = cms.Sequence( ALCARECOTkAlCosmicsCosmicTF0TTrackingDQM + ALCARECOTkAlCosmicsCosmicTF0TTkAlDQM )

#######################
### TkAlCosmicsRS0T ###
#######################
__selectionName = 'TkAlCosmicsRS0T'
ALCARECOTkAlCosmicsRS0TTrackingDQM = ALCARECOTkAlCosmicsCTF0TTrackingDQM.clone(
#names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName
)
ALCARECOTkAlCosmicsRS0TTkAlDQM = ALCARECOTkAlCosmicsCTF0TTkAlDQM.clone(
#names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    ReferenceTrackProducer = 'rsWithMaterialTracksP5',
    AlgoName = 'ALCARECO'+__selectionName
)
ALCARECOTkAlCosmicsRS0TDQM = cms.Sequence( ALCARECOTkAlCosmicsRS0TTrackingDQM + ALCARECOTkAlCosmicsRS0TTkAlDQM)

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
    AlgoName = 'ALCARECO'+__selectionName
)
ALCARECOTkAlCosmicsCosmicTFTkAlDQM = ALCARECOTkAlCosmicsCTF0TTkAlDQM.clone(
#names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    ReferenceTrackProducer = ALCARECOTkAlCosmicsCosmicTF0TTkAlDQM.ReferenceTrackProducer,
    AlgoName = 'ALCARECO'+__selectionName
)
ALCARECOTkAlCosmicsCosmicTFDQM = cms.Sequence( ALCARECOTkAlCosmicsCosmicTFTrackingDQM + ALCARECOTkAlCosmicsCosmicTFTkAlDQM)

#####################
### TkAlCosmicsRS ###
#####################
__selectionName = 'TkAlCosmicsRS'
ALCARECOTkAlCosmicsRSTrackingDQM = ALCARECOTkAlCosmicsCTFTrackingDQM.clone(
#names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName
)
ALCARECOTkAlCosmicsRSTkAlDQM = ALCARECOTkAlCosmicsCTF0TTkAlDQM.clone(
#names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    ReferenceTrackProducer = ALCARECOTkAlCosmicsRS0TTkAlDQM.ReferenceTrackProducer,
    AlgoName = 'ALCARECO'+__selectionName
)
ALCARECOTkAlCosmicsRSDQM = cms.Sequence( ALCARECOTkAlCosmicsRSTrackingDQM + ALCARECOTkAlCosmicsRSTkAlDQM )

