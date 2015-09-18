import FWCore.ParameterSet.Config as cms

##                      _              _       
##   ___ ___  _ __  ___| |_ __ _ _ __ | |_ ___ 
##  / __/ _ \| '_ \/ __| __/ _` | '_ \| __/ __|
## | (_| (_) | | | \__ \ || (_| | | | | |_\__ \
##  \___\___/|_| |_|___/\__\__,_|_| |_|\__|___/
##                                              

HLTPath = "HLT_Ele*"
HLTProcessName = "HLT"

#electron cuts
ELECTRON_ET_CUT_MIN = 10.0
TAG_ELECTRON_ET_CUT_MIN = 20.0
W_ELECTRON_ET_CUT_MIN = 27.0
ELECTRON_COLL = "gedGsfElectrons"
ELECTRON_CUTS = "(std::abs(obj.superCluster()->eta())<2.5) && (obj.ecalEnergy()*std::sin(obj.superClusterPosition().theta())>" + str(ELECTRON_ET_CUT_MIN) + ")"

#met, mt cuts for W selection
MET_CUT_MIN = 20.
MT_CUT_MIN = 50.

##    ____      __ _____ _           _                   
##   / ___|___ / _| ____| | ___  ___| |_ _ __ ___  _ __  
##  | |  _/ __| |_|  _| | |/ _ \/ __| __| '__/ _ \| '_ \ 
##  | |_| \__ \  _| |___| |  __/ (__| |_| | | (_) | | | |
##   \____|___/_| |_____|_|\___|\___|\__|_|  \___/|_| |_|
##  
#  GsfElectron ################ 
goodElectrons = cms.EDFilter("GsfElectronRefSelector",
    src = cms.InputTag( ELECTRON_COLL ),
    cut = cms.string( ELECTRON_CUTS )    
)

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
PassingWP90 = goodElectrons.clone(
cut = cms.string(
    goodElectrons.cut.value() +
    " && (obj.gsfTrack()->hitPattern().numberOfHits(reco::HitPattern::MISSING_INNER_HITS)<=1 && !(std::abs(obj.convDist()) < 0.02 && std::abs(obj.convDcot())<0.02))" #wrt std WP90 allowing 1 numberOfMissingExpectedHits 
    " && (obj.ecalEnergy()*std::sin(obj.superClusterPosition().theta())>" + str(ELECTRON_ET_CUT_MIN) + ")"
    " && ((obj.isEB()"
    " && ( obj.dr03TkSumPt()/obj.p4().Pt() <0.12 && obj.dr03EcalRecHitSumEt()/obj.p4().Pt() < 0.09 && obj.dr03HcalTowerSumEt()/obj.p4().Pt()  < 0.1 )"
    " && (obj.sigmaIetaIeta()<0.01)"
    " && ( std::abs(obj.deltaPhiSuperClusterTrackAtVtx())<0.8 )"
    " && ( std::abs(obj.deltaEtaSuperClusterTrackAtVtx())<0.007 )"
    " && (obj.hadronicOverEm()<0.12)"
    ")"
    " || (obj.isEE()"
    " && ( obj.dr03TkSumPt()/obj.p4().Pt() <0.07 && obj.dr03EcalRecHitSumEt()/obj.p4().P()t < 0.07 && obj.dr03HcalTowerSumEt()/obj.p4().Pt()  < 0.07 )"
    " && (obj.sigmaIetaIeta()<0.03)"
    " && ( std::abs(obj.deltaPhiSuperClusterTrackAtVtx())<0.7 )" 
    " && ( std::abs(obj.deltaEtaSuperClusterTrackAtVtx())<0.009 )"
    " && (obj.hadronicOverEm()<0.1) "
    "))"
    )
)

PassingWP80 = goodElectrons.clone(
cut = cms.string(
    goodElectrons.cut.value() +
    " && (obj.gsfTrack()->hitPattern().numberOfHits(reco::HitPattern::MISSING_INNER_HITS)==0 && !(std::abs(obj.convDist())<0.02 && std::abs(obj.convDcot())<0.02))" 
    " && (obj.ecalEnergy()*std::sin(obj.superClusterPosition().theta())>" + str(ELECTRON_ET_CUT_MIN) + ")"
    " && ((obj.isEB()"
    " && ( obj.dr03TkSumPt()/obj.p4().Pt() <0.12 && obj.dr03EcalRecHitSumEt()/obj.p4().Pt() < 0.09 && obj.dr03HcalTowerSumEt()/obj.p4().Pt()  < 0.1 )" #wrt std WP80 relaxing iso cuts to WP90 
    " && (obj.sigmaIetaIeta()<0.01)"
    " && ( std::abs(obj.deltaPhiSuperClusterTrackAtVtx())<0.06 )"
    " && ( std::abs(obj.deltaEtaSuperClusterTrackAtVtx())<0.004 )"
    " && (obj.hadronicOverEm()<0.12)"
    ")"
    " || (obj.isEE()"
    " && ( obj.dr03TkSumPt()/obj.p4().Pt() <0.05 && obj.dr03EcalRecHitSumEt()/obj.p4().Pt() < 0.06 && obj.dr03HcalTowerSumEt()/obj.p4().Pt()  < 0.03 )"
    " && (obj.sigmaIetaIeta()<0.03)"
    " && ( std::abs(obj.deltaPhiSuperClusterTrackAtVtx())<0.03 )" 
    " && ( std::abs(obj.deltaEtaSuperClusterTrackAtVtx())<0.007 )"
    " && (obj.hadronicOverEm()<0.10) "
    "))"
    )
) 

                         
##    _____     _                         __  __       _       _     _             
##   |_   _| __(_) __ _  __ _  ___ _ __  |  \/  | __ _| |_ ___| |__ (_)_ __   __ _ 
##     | || '__| |/ _` |/ _` |/ _ \ '__| | |\/| |/ _` | __/ __| '_ \| | '_ \ / _` |
##     | || |  | | (_| | (_| |  __/ |    | |  | | (_| | || (__| | | | | | | | (_| |
##     |_||_|  |_|\__, |\__, |\___|_|    |_|  |_|\__,_|\__\___|_| |_|_|_| |_|\__, |
##                |___/ |___/                                                |___/ 
##   
# Trigger  ##################
PassingHLT = cms.EDProducer("trgMatchGsfElectronProducer",    
    InputProducer = cms.InputTag( ELECTRON_COLL ),                          
    hltTags = cms.untracked.string( HLTPath ),
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
WElecTagHLT = PassingHLT.clone(
    InputProducer = cms.InputTag( "PassingWP80" )
    )

ele_sequence = cms.Sequence(
    goodElectrons +
    PassingWP80 + 
    WElecTagHLT
    )


##    _____ ___   ____    ____       _          
##   |_   _( _ ) |  _ \  |  _ \ __ _(_)_ __ ___ 
##     | | / _ \/\ |_) | | |_) / _` | | '__/ __|
##     | || (_>  <  __/  |  __/ (_| | | |  \__ \
##     |_| \___/\/_|     |_|   \__,_|_|_|  |___/
##                                              
##   

MT="sqrt(2*obj.daughter(0)->pt()*obj.daughter(1)->pt()*(1 - std::cos(obj.daughter(0)->phi() - obj.daughter(1)->phi())))"
elecMet = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("pfMet WElecTagHLT"), # charge coniugate states are implied
    checkCharge = cms.bool(False),                           
    cut   = cms.string(("obj.daughter(0)->pt() > %f && obj.daughter(1)->pt() > %f && "+MT+" > %f") % (MET_CUT_MIN, W_ELECTRON_ET_CUT_MIN, MT_CUT_MIN))
)
elecMetCounter = cms.EDFilter("CandViewCountFilter",
                                    src = cms.InputTag("elecMet"),
                                    minNumber = cms.uint32(1)
                                    )
elecMetFilter = cms.Sequence(elecMet * elecMetCounter)

import HLTrigger.HLTfilters.hltHighLevel_cfi
WEnuHltFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    throw = cms.bool(False),
    HLTPaths = [HLTPath]
    )

#--------------------------#
#recompute rho
import RecoJets.Configuration.RecoPFJets_cff
kt6PFJetsForRhoCorrectionWElectronSkim = RecoJets.Configuration.RecoPFJets_cff.kt6PFJets.clone(
    doRhoFastjet = True,
    Rho_EtaMax = 2.5
)


elecMetSeq = cms.Sequence( WEnuHltFilter * ele_sequence * elecMetFilter * kt6PFJetsForRhoCorrectionWElectronSkim)


from Configuration.EventContent.AlCaRecoOutput_cff import OutALCARECOEcalCalElectron
WElectronSkimContent = OutALCARECOEcalCalElectron.clone()
WElectronSkimContent.outputCommands.extend( [ 
  "keep *_pfMet_*_*", 
  "keep *_kt6*_rho_*", 
  "keep *_offlinePrimaryVerticesWithBS_*_*",
  "keep *_generator_*_*", 
  "keep *_rawDataCollector_*_*",
  'keep recoCaloClusters_*_*_*', 
  'keep recoPreshowerClusters_*_*_*',
  'keep *_reducedEcalRecHits*_*_*',
  'keep *_offlineBeamSpot_*_*',
  'keep *_allConversions_*_*',
  'keep *_gtDigis_*_*'
 ] )
