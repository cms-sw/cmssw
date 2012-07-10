import FWCore.ParameterSet.Config as cms

HLTPath = "HLT_Ele"
HLTProcessName = "HLT"

#electron cuts
ELECTRON_ET_CUT_MIN = 10.0
TAG_ELECTRON_ET_CUT_MIN = 20.0

#photon cuts (for probe)
PHOTON_ET_CUT_MIN = 10.0
PHOTON_COLL = "photons"
PHOTON_CUTS = "(abs(superCluster.eta)<2.5) && ( hadronicOverEm<0.5  ||  ((isEB && sigmaIetaIeta<0.02) || (isEE && sigmaIetaIeta<0.04)) ) && (superCluster.energy*sin(superCluster.position.theta)>" + str(PHOTON_ET_CUT_MIN) + ")"

#track cuts (for probe)
TRACK_ET_CUT_MIN = 25.0

#mass cuts (for T&P)
MASS_CUT_MIN = 30.
MASS_PHOTONTAG_CUT_MIN = 50.
MASS_TAGTRACK_CUT_MIN = 60.
MASS_TAGTRACK_CUT_MAX = 120.

##   ____                         ____ _           _            
##  / ___| _   _ _ __   ___ _ __ / ___| |_   _ ___| |_ ___ _ __ 
##  \___ \| | | | '_ \ / _ \ '__| |   | | | | / __| __/ _ \ '__|
##   ___) | |_| | |_) |  __/ |  | |___| | |_| \__ \ ||  __/ |   
##  |____/ \__,_| .__/ \___|_|   \____|_|\__,_|___/\__\___|_|   
##  

#  Photons!!! ################ 
goodPhotons = cms.EDFilter(
    "PhotonSelector",
    src = cms.InputTag( PHOTON_COLL ),
    cut = cms.string(PHOTON_CUTS)
    )

photon_sequence = cms.Sequence(
    goodPhotons
    )


# Tracks ###########
from  PhysicsTools.RecoAlgos.allTrackCandidates_cfi import  allTrackCandidates

goodTracks = cms.EDFilter("CandViewRefSelector",
    filter = cms.bool(True),
    src = cms.InputTag("allTrackCandidates"),
    cut = cms.string('pt > '+str(PHOTON_ET_CUT_MIN))
)

track_sequence = cms.Sequence(allTrackCandidates + goodTracks)

##    ____      __ _____ _           _                   
##   / ___|___ / _| ____| | ___  ___| |_ _ __ ___  _ __  
##  | |  _/ __| |_|  _| | |/ _ \/ __| __| '__/ _ \| '_ \ 
##  | |_| \__ \  _| |___| |  __/ (__| |_| | | (_) | | | |
##   \____|___/_| |_____|_|\___|\___|\__|_|  \___/|_| |_|
##  
#  GsfElectron ################
from DPGAnalysis.Skims.WElectronSkim_cff import goodElectrons


GsfMatchedPhotonCands = cms.EDProducer("ElectronMatchedCandidateProducer",
   src     = cms.InputTag("goodPhotons"),
   ReferenceElectronCollection = cms.untracked.InputTag("goodElectrons"),
   deltaR =  cms.untracked.double(0.3)
)

##    _____ _           _                     ___    _ 
##   | ____| | ___  ___| |_ _ __ ___  _ __   |_ _|__| |
##   |  _| | |/ _ \/ __| __| '__/ _ \| '_ \   | |/ _` |
##   | |___| |  __/ (__| |_| | | (_) | | | |  | | (_| |
##   |_____|_|\___|\___|\__|_|  \___/|_| |_| |___\__,_|
##   
# Electron ID  ######
from DPGAnalysis.Skims.WElectronSkim_cff import PassingWP90
                         
##    _____     _                         __  __       _       _     _             
##   |_   _| __(_) __ _  __ _  ___ _ __  |  \/  | __ _| |_ ___| |__ (_)_ __   __ _ 
##     | || '__| |/ _` |/ _` |/ _ \ '__| | |\/| |/ _` | __/ __| '_ \| | '_ \ / _` |
##     | || |  | | (_| | (_| |  __/ |    | |  | | (_| | || (__| | | | | | | | (_| |
##     |_||_|  |_|\__, |\__, |\___|_|    |_|  |_|\__,_|\__\___|_| |_|_|_| |_|\__, |
##                |___/ |___/                                                |___/ 
##   
# Trigger  ##################
PassingHLT = cms.EDProducer("trgMatchGsfElectronProducer",    
    InputProducer = cms.InputTag( 'gsfElectrons' ),
    hltTags = cms.untracked.string("HLT_Ele"),
    triggerEventTag = cms.untracked.InputTag("hltTriggerSummaryAOD","",HLTProcessName),
    triggerResultsTag = cms.untracked.InputTag("TriggerResults","",HLTProcessName)   
)

##    _____             ____        __ _       _ _   _             
##   |_   _|_ _  __ _  |  _ \  ___ / _(_)_ __ (_) |_(_) ___  _ __  
##     | |/ _` |/ _` | | | | |/ _ \ |_| | '_ \| | __| |/ _ \| '_ \ 
##     | | (_| | (_| | | |_| |  __/  _| | | | | | |_| | (_) | | | |
##     |_|\__,_|\__, | |____/ \___|_| |_|_| |_|_|\__|_|\___/|_| |_|
##              |___/
## 
ZElecTagHLT = PassingHLT.clone(
    InputProducer = cms.InputTag( "PassingWP90" )
    )

Zele_sequence = cms.Sequence(
    goodElectrons +
    PassingWP90 +
    ZElecTagHLT
    )


##    _____ ___   ____    ____       _          
##   |_   _( _ ) |  _ \  |  _ \ __ _(_)_ __ ___ 
##     | | / _ \/\ |_) | | |_) / _` | | '__/ __|
##     | || (_>  <  __/  |  __/ (_| | | |  \__ \
##     |_| \___/\/_|     |_|   \__,_|_|_|  |___/
##                                              
##   
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
ZEEHltFilter = copy.deepcopy(hltHighLevel)
ZEEHltFilter.throw = cms.bool(False)
ZEEHltFilter.HLTPaths = ["HLT_Ele*"]

tagPhoton = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("ZElecTagHLT goodPhotons"), # charge coniugate states are implied
    checkCharge = cms.bool(False),                           
    cut   = cms.string("mass > " + str(MASS_PHOTONTAG_CUT_MIN))
)
tagPhotonCounter = cms.EDFilter("CandViewCountFilter",
                                    src = cms.InputTag("tagPhoton"),
                                    minNumber = cms.uint32(1)
                                    )
tagPhotonFilter = cms.Sequence(tagPhoton * tagPhotonCounter)
tagPhotonSeq = cms.Sequence( ZEEHltFilter *(photon_sequence + Zele_sequence) * tagPhotonFilter )

tagTrack = tagPhoton.clone(
    decay = cms.string("ZElecTagHLT goodTracks"),
    cut   = cms.string("mass > " + str(MASS_TAGTRACK_CUT_MIN)+ ' && mass < ' + str(MASS_TAGTRACK_CUT_MAX)),
)
tagTrackCounter = tagPhotonCounter.clone(
    src = cms.InputTag("tagTrack")
    )
tagTrackFilter = cms.Sequence(tagTrack * tagTrackCounter)
tagTrackSeq = cms.Sequence( ZEEHltFilter * (track_sequence + Zele_sequence) * tagTrackFilter )

tagGsf = tagPhoton.clone(
    decay = cms.string("PassingWP90 goodElectrons")
    )
tagGsfCounter = tagPhotonCounter.clone(
    src = cms.InputTag("tagGsf")
    )
tagGsfFilter = cms.Sequence(tagGsf * tagGsfCounter)
tagGsfSeq = cms.Sequence( ZEEHltFilter * (Zele_sequence) * tagGsfFilter )  
