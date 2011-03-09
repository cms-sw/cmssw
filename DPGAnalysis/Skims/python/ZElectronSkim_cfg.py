import FWCore.ParameterSet.Config as cms

##                      _              _       
##   ___ ___  _ __  ___| |_ __ _ _ __ | |_ ___ 
##  / __/ _ \| '_ \/ __| __/ _` | '_ \| __/ __|
## | (_| (_) | | | \__ \ || (_| | | | | |_\__ \
##  \___\___/|_| |_|___/\__\__,_|_| |_|\__|___/
##                                              

GLOBAL_TAG = 'GR_R_39X_V4::All'
HLTPath = "HLT_Ele"
HLTProcessName = "HLT"

#electron cuts
ELECTRON_ET_CUT_MIN = 10.0
TAG_ELECTRON_ET_CUT_MIN = 20.0
ELECTRON_COLL = "gsfElectrons"
ELECTRON_CUTS = "(abs(superCluster.eta)<2.5) && (ecalEnergy*sin(superClusterPosition.theta)>" + str(ELECTRON_ET_CUT_MIN) + ")"

#photon cuts (for probe)
PHOTON_ET_CUT_MIN = 10.0
PHOTON_COLL = "photons"
PHOTON_CUTS = "hadronicOverEm<0.2 && (abs(superCluster.eta)<2.5)  && ((isEB && sigmaIetaIeta<0.015) || (isEE && sigmaIetaIeta<0.035)) && (superCluster.energy*sin(superCluster.position.theta)>" + str(PHOTON_ET_CUT_MIN) + ")"

#track cuts (for probe)
TRACK_ET_CUT_MIN = 25.0

#mass cuts (for T&P)
MASS_CUT_MIN = 30.
MASS_TAGTRACK_CUT_MIN = 60.
MASS_TAGTRACK_CUT_MAX = 120.

##    ___            _           _      
##   |_ _|_ __   ___| |_   _  __| | ___ 
##    | || '_ \ / __| | | | |/ _` |/ _ \
##    | || | | | (__| | |_| | (_| |  __/
##   |___|_| |_|\___|_|\__,_|\__,_|\___|
##
process = cms.Process("ZEESKIM")
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load("Configuration.StandardSequences.Geometry_cff")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = GLOBAL_TAG

process.load('Configuration/EventContent/EventContent_cff')
process.FEVTEventContent.outputCommands.append("drop *_MEtoEDMConverter_*_*")
process.FEVTEventContent.outputCommands.append("drop *_*_*_SKIM")

process.options = cms.untracked.PSet(
 wantSummary = cms.untracked.bool(True)
)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.MessageLogger.cerr.FwkReport.reportEvery = 1000

##   ____             _ ____                           
##  |  _ \ ___   ___ | / ___|  ___  _   _ _ __ ___ ___ 
##  | |_) / _ \ / _ \| \___ \ / _ \| | | | '__/ __/ _ \
##  |  __/ (_) | (_) | |___) | (_) | |_| | | | (_|  __/
##  |_|   \___/ \___/|_|____/ \___/ \__,_|_|  \___\___|
##  
process.source = cms.Source("PoolSource", 
    fileNames = cms.untracked.vstring(
    '/store/data/Run2010B/Electron/RECO/Dec22ReReco_v1/0012/5C58C005-C70E-E011-9541-0018F3D095FC.root'       
    ),
                            secondaryFileNames = cms.untracked.vstring(
'/store/data/Run2010B/Electron/RAW/v1/000/148/864/780B9035-36E0-DF11-8A3A-000423D98950.root',
'/store/data/Run2010B/Electron/RAW/v1/000/148/864/2019FF51-45E0-DF11-945D-00304879FC6C.root',
'/store/data/Run2010B/Electron/RAW/v1/000/148/864/44B7C28B-30E0-DF11-960E-001617E30E28.root',
'/store/data/Run2010B/Electron/RAW/v1/000/148/864/4890366C-2CE0-DF11-9804-001617DBCF90.root',
'/store/data/Run2010B/Electron/RAW/v1/000/148/864/684C5EBC-2DE0-DF11-B38F-000423D996C8.root',
'/store/data/Run2010B/Electron/RAW/v1/000/148/864/780B9035-36E0-DF11-8A3A-000423D98950.root',
'/store/data/Run2010B/Electron/RAW/v1/000/148/864/7E9A8843-31E0-DF11-AE62-001617E30D4A.root',
'/store/data/Run2010B/Electron/RAW/v1/000/148/864/AEE6EF34-34E0-DF11-AF76-0019B9F72BFF.root',
'/store/data/Run2010B/Electron/RAW/v1/000/148/864/B0843493-37E0-DF11-A2AC-000423D996C8.root',
'/store/data/Run2010B/Electron/RAW/v1/000/148/864/DCFF7CA0-2BE0-DF11-93B9-000423D996C8.root',
'/store/data/Run2010B/Electron/RAW/v1/000/148/864/E0954CE4-32E0-DF11-8546-000423D94908.root',
'/store/data/Run2010B/Electron/RAW/v1/000/148/864/E87EF327-2FE0-DF11-BD11-001617E30D4A.root'
)
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )    
process.source.inputCommands = cms.untracked.vstring("keep *","drop *_MEtoEDMConverter_*_*")

##   ____                         ____ _           _            
##  / ___| _   _ _ __   ___ _ __ / ___| |_   _ ___| |_ ___ _ __ 
##  \___ \| | | | '_ \ / _ \ '__| |   | | | | / __| __/ _ \ '__|
##   ___) | |_| | |_) |  __/ |  | |___| | |_| \__ \ ||  __/ |   
##  |____/ \__,_| .__/ \___|_|   \____|_|\__,_|___/\__\___|_|   
##  

#  Photons!!! ################ 
process.goodPhotons = cms.EDFilter(
    "PhotonSelector",
    src = cms.InputTag( PHOTON_COLL ),
    cut = cms.string(PHOTON_CUTS)
    )

process.photon_sequence = cms.Sequence(
    process.goodPhotons
    )


# Tracks ###########
process.load("PhysicsTools.RecoAlgos.allTrackCandidates_cfi")

process.goodTracks = cms.EDFilter("CandViewRefSelector",
    filter = cms.bool(True),
    src = cms.InputTag("allTrackCandidates"),
    cut = cms.string('pt > '+str(PHOTON_ET_CUT_MIN))
)

process.track_sequence = cms.Sequence(process.allTrackCandidates + process.goodTracks)

##    ____      __ _____ _           _                   
##   / ___|___ / _| ____| | ___  ___| |_ _ __ ___  _ __  
##  | |  _/ __| |_|  _| | |/ _ \/ __| __| '__/ _ \| '_ \ 
##  | |_| \__ \  _| |___| |  __/ (__| |_| | | (_) | | | |
##   \____|___/_| |_____|_|\___|\___|\__|_|  \___/|_| |_|
##  
#  GsfElectron ################ 
process.goodElectrons = cms.EDFilter("GsfElectronRefSelector",
    src = cms.InputTag( ELECTRON_COLL ),
    cut = cms.string( ELECTRON_CUTS )    
)

process.GsfMatchedPhotonCands = cms.EDProducer("ElectronMatchedCandidateProducer",
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
process.PassingWP90 = process.goodElectrons.clone()
process.PassingWP90.cut = cms.string(
    process.goodElectrons.cut.value() +
    " && (gsfTrack.trackerExpectedHitsInner.numberOfHits<=1 && !(-0.02<convDist<0.02 && -0.02<convDcot<0.02))" #wrt std WP90 allowing 1 numberOfMissingExpectedHits 
    " && (ecalEnergy*sin(superClusterPosition.theta)>" + str(ELECTRON_ET_CUT_MIN) + ")"
    " && ((isEB"
    " && ( dr03TkSumPt/p4.Pt <0.12 && dr03EcalRecHitSumEt/p4.Pt < 0.09 && dr03HcalTowerSumEt/p4.Pt  < 0.1 )"
    " && (sigmaIetaIeta<0.01)"
    " && ( -0.8<deltaPhiSuperClusterTrackAtVtx<0.8 )"
    " && ( -0.007<deltaEtaSuperClusterTrackAtVtx<0.007 )"
    " && (hadronicOverEm<0.12)"
    ")"
    " || (isEE"
    " && ( dr03TkSumPt/p4.Pt <0.05 && dr03EcalRecHitSumEt/p4.Pt < 0.06 && dr03HcalTowerSumEt/p4.Pt  < 0.03 )"
    " && (sigmaIetaIeta<0.03)"
    " && ( -0.7<deltaPhiSuperClusterTrackAtVtx<0.7 )" 
    " && ( -0.009<deltaEtaSuperClusterTrackAtVtx<0.009 )"
    " && (hadronicOverEm<0.05) "
    "))"
    ) 

                         
##    _____     _                         __  __       _       _     _             
##   |_   _| __(_) __ _  __ _  ___ _ __  |  \/  | __ _| |_ ___| |__ (_)_ __   __ _ 
##     | || '__| |/ _` |/ _` |/ _ \ '__| | |\/| |/ _` | __/ __| '_ \| | '_ \ / _` |
##     | || |  | | (_| | (_| |  __/ |    | |  | | (_| | || (__| | | | | | | | (_| |
##     |_||_|  |_|\__, |\__, |\___|_|    |_|  |_|\__,_|\__\___|_| |_|_|_| |_|\__, |
##                |___/ |___/                                                |___/ 
##   
# Trigger  ##################
process.PassingHLT = cms.EDProducer("trgMatchGsfElectronProducer",    
    InputProducer = cms.InputTag( ELECTRON_COLL ),                          
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
process.TagHLT = process.PassingHLT.clone()
#process.Tag = process.PassingWP90.clone()
process.TagHLT.InputProducer = cms.InputTag( "PassingWP90" )

process.ele_sequence = cms.Sequence(
    process.goodElectrons +
    process.PassingWP90 +
 #   process.PassingHLT +
    process.TagHLT
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
process.ZEEHltFilter = copy.deepcopy(hltHighLevel)
process.ZEEHltFilter.throw = cms.bool(False)
process.ZEEHltFilter.HLTPaths = ["HLT_Ele*"]

process.tagPhoton = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("TagHLT goodPhotons"), # charge coniugate states are implied
    checkCharge = cms.bool(False),                           
    cut   = cms.string("mass > " + str(MASS_CUT_MIN))
)
process.tagPhotonCounter = cms.EDFilter("CandViewCountFilter",
                                    src = cms.InputTag("tagPhoton"),
                                    minNumber = cms.uint32(1)
                                    )
process.tagPhotonFilter = cms.Sequence(process.tagPhoton * process.tagPhotonCounter)
process.tagPhotonPath = cms.Path( process.ZEEHltFilter *(process.photon_sequence + process.ele_sequence) * process.tagPhotonFilter )

process.tagTrack = process.tagPhoton.clone()
process.tagTrack.decay = cms.string("TagHLT goodTracks")
process.tagTrack.cut   = cms.string("mass > " + str(MASS_TAGTRACK_CUT_MIN)+ ' && mass < ' + str(MASS_TAGTRACK_CUT_MAX))
process.tagTrackCounter = process.tagPhotonCounter.clone()
process.tagTrackCounter.src = cms.InputTag("tagTrack")
process.tagTrackFilter = cms.Sequence(process.tagTrack * process.tagTrackCounter)
process.tagTrackPath = cms.Path( process.ZEEHltFilter * (process.track_sequence + process.ele_sequence) * process.tagTrackFilter )

process.tagGsf = process.tagPhoton.clone()
process.tagGsf.decay = cms.string("PassingWP90 goodElectrons")
process.tagGsfCounter = process.tagPhotonCounter.clone()
process.tagGsfCounter.src = cms.InputTag("tagGsf")
process.tagGsfFilter = cms.Sequence(process.tagGsf * process.tagGsfCounter)
process.tagGsfPath = cms.Path( process.ZEEHltFilter * (process.ele_sequence) * process.tagGsfFilter )  

process.outZfilter = cms.OutputModule("PoolOutputModule",
                                       # splitLevel = cms.untracked.int32(0),
                                       outputCommands = process.FEVTEventContent.outputCommands,
                                       fileName = cms.untracked.string('EGM_Z_filter.root'),
                                       dataset = cms.untracked.PSet(dataTier = cms.untracked.string('RAW-RECO'),
                                                                    filterName = cms.untracked.string('EGMZFilter')),
                                       SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('tagPhotonPath','tagTrackPath','tagGsfPath')
                                                                         ))


#======================

process.outpath = cms.EndPath(process.outZfilter)
