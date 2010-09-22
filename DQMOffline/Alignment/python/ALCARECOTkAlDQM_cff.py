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

from DQM.HLTEvF.HLTMonBitSummary_cfi import hltMonBitSummary
from Alignment.CommonAlignmentProducer.ALCARECOTkAlZMuMu_cff import ALCARECOTkAlZMuMuHLT
ALCARECOTkAlZMuMuHLTDQM = hltMonBitSummary.clone(
    directory = "AlCaReco/"+__selectionName+"/HLTSummary",
    histLabel = __selectionName,
    HLTPaths = ["HLT_.*Mu.*"],
    eventSetupPathsKey =  ALCARECOTkAlZMuMuHLT.eventSetupPathsKey.value()
)

ALCARECOTkAlZMuMuDQM = cms.Sequence( ALCARECOTkAlZMuMuTrackingDQM + ALCARECOTkAlZMuMuTkAlDQM + ALCARECOTkAlZMuMuHLTDQM )

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

from Alignment.CommonAlignmentProducer.ALCARECOTkAlJpsiMuMu_cff import ALCARECOTkAlJpsiMuMuHLT
ALCARECOTkAlJpsiMuMuHLTDQM = hltMonBitSummary.clone(
    directory = "AlCaReco/"+__selectionName+"/HLTSummary",
    histLabel = __selectionName,
    HLTPaths = ["HLT_.*Mu.*"],
    eventSetupPathsKey =  ALCARECOTkAlJpsiMuMuHLT.eventSetupPathsKey.value()
)

ALCARECOTkAlJpsiMuMuDQM = cms.Sequence( ALCARECOTkAlJpsiMuMuTrackingDQM + ALCARECOTkAlJpsiMuMuTkAlDQM + ALCARECOTkAlJpsiMuMuHLTDQM)

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

from Alignment.CommonAlignmentProducer.ALCARECOTkAlUpsilonMuMu_cff import ALCARECOTkAlUpsilonMuMuHLT
ALCARECOTkAlUpsilonMuMuHLTDQM = hltMonBitSummary.clone(
    directory = "AlCaReco/"+__selectionName+"/HLTSummary",
    histLabel = __selectionName,
    HLTPaths = ["HLT_.*Mu.*"],
    eventSetupPathsKey =  ALCARECOTkAlUpsilonMuMuHLT.eventSetupPathsKey.value()
)

ALCARECOTkAlUpsilonMuMuDQM = cms.Sequence( ALCARECOTkAlUpsilonMuMuTrackingDQM + ALCARECOTkAlUpsilonMuMuTkAlDQM + ALCARECOTkAlUpsilonMuMuHLTDQM)

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

# no BeamHalo HLT bits yet
#from Alignment.CommonAlignmentProducer.ALCARECOTkAlBeamHalo_cff import ALCARECOTkAlBeamHaloHLT
#ALCARECOTkAlBeamHaloHLTDQM = hltMonBitSummary.clone(
#    directory = "AlCaReco/"+__selectionName+"/HLTSummary",
#    histLabel = __selectionName,
#    HLTPaths = ["HLT_.*L1.*"],
#    eventSetupPathsKey =  ALCARECOTkAlBeamHaloHLT.eventSetupPathsKey.value()
#)

ALCARECOTkAlBeamHaloDQM = cms.Sequence( ALCARECOTkAlBeamHaloTrackingDQM 
#+ ALCARECOTkAlBeamHaloHLTDQM 
)

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

from Alignment.CommonAlignmentProducer.ALCARECOTkAlMinBias_cff import ALCARECOTkAlMinBiasNOTHLT
ALCARECOTkAlMinBiasNOTHLTDQM = hltMonBitSummary.clone(
    directory = "AlCaReco/"+__selectionName+"/HLTSummaryNOT",
    histLabel = __selectionName,
    HLTPaths = ["HLT_.*L1.*"],
    eventSetupPathsKey =  ALCARECOTkAlMinBiasNOTHLT.eventSetupPathsKey.value()
)

from Alignment.CommonAlignmentProducer.ALCARECOTkAlMinBias_cff import ALCARECOTkAlMinBiasHLT
ALCARECOTkAlMinBiasHLTDQM = hltMonBitSummary.clone(
    directory = "AlCaReco/"+__selectionName+"/HLTSummary",
    histLabel = __selectionName,
    HLTPaths = [],
    eventSetupPathsKey =  ALCARECOTkAlMinBiasHLT.eventSetupPathsKey.value()
)

ALCARECOTkAlMinBiasDQM = cms.Sequence( ALCARECOTkAlMinBiasTrackingDQM + ALCARECOTkAlMinBiasTkAlDQM+ALCARECOTkAlMinBiasHLTDQM+ALCARECOTkAlMinBiasNOTHLTDQM)

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
# margins and settings
    TkSizeBin = 71,
    TkSizeMin = -0.5,
    TkSizeMax = 70.5,
    TrackPtMax = 30
)

ALCARECOTkAlMinBiasHITkAlDQM = ALCARECOTkAlMinBiasTkAlDQM.clone(
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

from Alignment.CommonAlignmentProducer.ALCARECOTkAlMinBiasHI_cff import ALCARECOTkAlMinBiasHIHLT
ALCARECOTkAlMinBiasHIHLTDQM = hltMonBitSummary.clone(
    directory = "AlCaReco/"+__selectionName+"/HLTSummary",
    histLabel = __selectionName,
    HLTPaths = [],
    eventSetupPathsKey =  ALCARECOTkAlMinBiasHIHLT.eventSetupPathsKey.value()
)

ALCARECOTkAlMinBiasHIDQM = cms.Sequence( ALCARECOTkAlMinBiasHITrackingDQM + ALCARECOTkAlMinBiasHITkAlDQM+ALCARECOTkAlMinBiasHIHLTDQM)

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

from Alignment.CommonAlignmentProducer.ALCARECOTkAlMuonIsolated_cff import ALCARECOTkAlMuonIsolatedHLT
ALCARECOTkAlMuonIsolatedHLTDQM = hltMonBitSummary.clone(
    directory = "AlCaReco/"+__selectionName+"/HLTSummary",
    histLabel = __selectionName,
    HLTPaths = ["HLT_.*L1.*"],
    eventSetupPathsKey =  ALCARECOTkAlMuonIsolatedHLT.eventSetupPathsKey.value()
)

ALCARECOTkAlMuonIsolatedDQM = cms.Sequence( ALCARECOTkAlMuonIsolatedTrackingDQM + ALCARECOTkAlMuonIsolatedTkAlDQM+ALCARECOTkAlMuonIsolatedHLTDQM)

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
    ReferenceTrackProducer = 'regionalCosmicTracks',
    AlgoName = 'ALCARECO'+__selectionName
)
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmicsInCollisions_cff import ALCARECOTkAlCosmicsInCollisionsHLT
ALCARECOTkAlCosmicsInCollisionsHLTDQM = hltMonBitSummary.clone(
    directory = "AlCaReco/"+__selectionName+"/HLTSummary",
    histLabel = __selectionName,
    HLTPaths = ["HLT_.*L1.*"],
    eventSetupPathsKey =  ALCARECOTkAlCosmicsInCollisionsHLT.eventSetupPathsKey.value()
)
ALCARECOTkAlCosmicsInCollisionsDQM = cms.Sequence( ALCARECOTkAlCosmicsInCollisionsTrackingDQM + ALCARECOTkAlCosmicsInCollisionsTkAlDQM +ALCARECOTkAlCosmicsInCollisionsHLTDQM)

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
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmics0THLT_cff import ALCARECOTkAlCosmics0THLT
ALCARECOTkAlCosmicsCTF0THLTDQM = hltMonBitSummary.clone(
    directory = "AlCaReco/"+__selectionName+"/HLTSummary",
    histLabel = __selectionName,
    HLTPaths = ["HLT_.*L1.*"],
    eventSetupPathsKey =  ALCARECOTkAlCosmics0THLT.eventSetupPathsKey.value()
)
ALCARECOTkAlCosmicsCTF0TDQM = cms.Sequence( ALCARECOTkAlCosmicsCTF0TTrackingDQM + ALCARECOTkAlCosmicsCTF0TTkAlDQM+ALCARECOTkAlCosmicsCTF0THLTDQM)

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
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmics0THLT_cff import ALCARECOTkAlCosmics0THLT
ALCARECOTkAlCosmicsCosmicTF0THLTDQM = hltMonBitSummary.clone(
    directory = "AlCaReco/"+__selectionName+"/HLTSummary",
    histLabel = __selectionName,
    HLTPaths = ["HLT_.*L1.*"],
    eventSetupPathsKey =  ALCARECOTkAlCosmics0THLT.eventSetupPathsKey.value()
)
ALCARECOTkAlCosmicsCosmicTF0TDQM = cms.Sequence( ALCARECOTkAlCosmicsCosmicTF0TTrackingDQM + ALCARECOTkAlCosmicsCosmicTF0TTkAlDQM +ALCARECOTkAlCosmicsCosmicTF0THLTDQM)

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
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmicsHLT_cff import ALCARECOTkAlCosmicsHLT
ALCARECOTkAlCosmicsCTFHLTDQM = hltMonBitSummary.clone(
    directory = "AlCaReco/"+__selectionName+"/HLTSummary",
    histLabel = __selectionName,
    HLTPaths = ["HLT_.*L1.*"],
    eventSetupPathsKey =  ALCARECOTkAlCosmicsHLT.eventSetupPathsKey.value()
)
ALCARECOTkAlCosmicsCTFDQM = cms.Sequence( ALCARECOTkAlCosmicsCTFTrackingDQM + ALCARECOTkAlCosmicsCTFTkAlDQM +ALCARECOTkAlCosmicsCTFHLTDQM)

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
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmicsHLT_cff import ALCARECOTkAlCosmicsHLT
ALCARECOTkAlCosmicsCosmicTFHLTDQM = hltMonBitSummary.clone(
    directory = "AlCaReco/"+__selectionName+"/HLTSummary",
    histLabel = __selectionName,
    HLTPaths = ["HLT_.*L1.*"],
    eventSetupPathsKey =  ALCARECOTkAlCosmicsHLT.eventSetupPathsKey.value()
)
ALCARECOTkAlCosmicsCosmicTFDQM = cms.Sequence( ALCARECOTkAlCosmicsCosmicTFTrackingDQM + ALCARECOTkAlCosmicsCosmicTFTkAlDQM+ALCARECOTkAlCosmicsCosmicTFHLTDQM)
