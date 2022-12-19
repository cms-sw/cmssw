## /*****************************************************************************
##  * Project: CMS detector at the CERN
##  *
##  * Package: PhysicsTools/TagAndProbe
##  *
##  *
##  * Authors:
##  *
##  *   Kalanand Mishra, Fermilab - kalanand@fnal.gov
##  *
##  * Description:
##  *   - Produces tag & probe TTree for further analysis and computing efficiency
##  *
##  * History:
##  *   
##  * 
##  *****************************************************************************/


import FWCore.ParameterSet.Config as cms

##                      _              _       
##   ___ ___  _ __  ___| |_ __ _ _ __ | |_ ___ 
##  / __/ _ \| '_ \/ __| __/ _` | '_ \| __/ __|
## | (_| (_) | | | \__ \ || (_| | | | | |_\__ \
##  \___\___/|_| |_|___/\__\__,_|_| |_|\__|___/
##                                              
################################################
MC_flag = True
GLOBAL_TAG = 'GR_R_42_V12::All'
if MC_flag:
    GLOBAL_TAG = 'START42_V12::All'
    
HLTPath = "HLT_Ele52_CaloIdVT_TrkIdT_v3"
HLTProcessName = "HLT"
if MC_flag:
    HLTPath = "HLT_Ele32_SW_TighterEleId_L1R_v2"
    HLTProcessName = "HLT"

OUTPUT_FILE_NAME = "testNewWrite.root"


ELECTRON_ET_CUT_MIN = 17.0
ELECTRON_COLL = "gsfElectrons"
ELECTRON_CUTS = "ecalDrivenSeed==1 && (abs(superCluster.eta)<2.5) && !(1.4442<abs(superCluster.eta)<1.566) && (ecalEnergy*sin(superClusterPosition.theta)>" + str(ELECTRON_ET_CUT_MIN) + ")"
####

PHOTON_COLL = "photons"
PHOTON_CUTS = "hadronicOverEm<0.15 && (abs(superCluster.eta)<2.5) && !(1.4442<abs(superCluster.eta)<1.566) && ((isEB && sigmaIetaIeta<0.01) || (isEE && sigmaIetaIeta<0.03)) && (superCluster.energy*sin(superCluster.position.theta)>" + str(ELECTRON_ET_CUT_MIN) + ")"
####

SUPERCLUSTER_COLL_EB = "hybridSuperClusters"
SUPERCLUSTER_COLL_EE = "multi5x5SuperClustersWithPreshower"
if MC_flag:
    SUPERCLUSTER_COLL_EB = "correctedHybridSuperClusters"
    SUPERCLUSTER_COLL_EE = "correctedMulti5x5SuperClustersWithPreshower"
SUPERCLUSTER_CUTS = "abs(eta)<2.5 && !(1.4442< abs(eta) <1.566) && et>" + str(ELECTRON_ET_CUT_MIN)


JET_COLL = "ak5PFJets"
JET_CUTS = "abs(eta)<2.6 && chargedHadronEnergyFraction>0 && electronEnergyFraction<0.1 && nConstituents>1 && neutralHadronEnergyFraction<0.99 && neutralEmEnergyFraction<0.99" 
########################

##    ___            _           _      
##   |_ _|_ __   ___| |_   _  __| | ___ 
##    | || '_ \ / __| | | | |/ _` |/ _ \
##    | || | | | (__| | |_| | (_| |  __/
##   |___|_| |_|\___|_|\__,_|\__,_|\___|
##
process = cms.Process("TagProbe")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.GlobalTag.globaltag = GLOBAL_TAG
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
#process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.MessageLogger.cerr.FwkReport.reportEvery = 1000



############# Needed for pileup re-weighting ##########
process.pileupReweightingProducer = cms.EDProducer("PileupWeightProducer",
                                                   FirstTime = cms.untracked.bool(True),
                                                   PileupMCFile = cms.untracked.string("PUMC_dist_flat10.root"),
                                                   PileupDataFile = cms.untracked.string("PUData_finebin_dist.root")
)
#########
##   ____             _ ____                           
##  |  _ \ ___   ___ | / ___|  ___  _   _ _ __ ___ ___ 
##  | |_) / _ \ / _ \| \___ \ / _ \| | | | '__/ __/ _ \
##  |  __/ (_) | (_) | |___) | (_) | |_| | | | (_|  __/
##  |_|   \___/ \___/|_|____/ \___/ \__,_|_|  \___\___|
##  
process.source = cms.Source("PoolSource", 
    fileNames = cms.untracked.vstring(

##        '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v4/000/167/283/BC0BDDDF-829D-E011-89EE-0030487CF41E.root',
##        '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v4/000/167/282/F0A7CE51-AE9D-E011-9C9E-003048D374F2.root',
##        '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v4/000/167/282/B2D25F4E-A79D-E011-94AD-003048F024DC.root',
##        '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v4/000/167/282/A89EC329-979D-E011-92F3-001D09F24DDF.root',
##        '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v4/000/167/282/5831779B-989D-E011-995A-0019B9F730D2.root',
##        '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v4/000/167/282/5021FAFB-AE9D-E011-B07F-003048D37538.root',
##        '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v4/000/167/282/34B74172-A99D-E011-8CD1-003048D2BDD8.root',

       '/store/mc/Summer11/ZJetToEE_Pt-30to50_TuneZ2_7TeV_pythia6/AODSIM/PU_S3_START42_V11-v2/0000/F6082563-6C7F-E011-9730-00215E222790.root',
       '/store/mc/Summer11/ZJetToEE_Pt-30to50_TuneZ2_7TeV_pythia6/AODSIM/PU_S3_START42_V11-v2/0000/F4B4B87C-927F-E011-9805-00215E21D786.root',
       '/store/mc/Summer11/ZJetToEE_Pt-30to50_TuneZ2_7TeV_pythia6/AODSIM/PU_S3_START42_V11-v2/0000/E673A7C3-927F-E011-B8C0-00215E2205AC.root',
       '/store/mc/Summer11/ZJetToEE_Pt-30to50_TuneZ2_7TeV_pythia6/AODSIM/PU_S3_START42_V11-v2/0000/D0725463-6C7F-E011-9285-00215E222790.root',
       '/store/mc/Summer11/ZJetToEE_Pt-30to50_TuneZ2_7TeV_pythia6/AODSIM/PU_S3_START42_V11-v2/0000/BC3E4618-BC80-E011-8AB8-E41F13181A50.root',
       '/store/mc/Summer11/ZJetToEE_Pt-30to50_TuneZ2_7TeV_pythia6/AODSIM/PU_S3_START42_V11-v2/0000/A60D2563-6C7F-E011-84A1-00215E222790.root',


       
    )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )    
process.source.inputCommands = cms.untracked.vstring("keep *","drop *_MEtoEDMConverter_*_*")

##   ____                         ____ _           _            
##  / ___| _   _ _ __   ___ _ __ / ___| |_   _ ___| |_ ___ _ __ 
##  \___ \| | | | '_ \ / _ \ '__| |   | | | | / __| __/ _ \ '__|
##   ___) | |_| | |_) |  __/ |  | |___| | |_| \__ \ ||  __/ |   
##  |____/ \__,_| .__/ \___|_|   \____|_|\__,_|___/\__\___|_|   
##  

#  SuperClusters  ################
process.superClusters = cms.EDProducer("SuperClusterMerger",
   src = cms.VInputTag(cms.InputTag( SUPERCLUSTER_COLL_EB ,"", "RECO"),
                       cms.InputTag( SUPERCLUSTER_COLL_EE ,"", "RECO") )  
)

process.superClusterCands = cms.EDProducer("ConcreteEcalCandidateProducer",
   src = cms.InputTag("superClusters"),
   particleType = cms.int32(11),
)

#   Get the above SC's Candidates and place a cut on their Et and eta
process.goodSuperClusters = cms.EDFilter("CandViewSelector",
      src = cms.InputTag("superClusterCands"),
      cut = cms.string( SUPERCLUSTER_CUTS ),
      filter = cms.bool(True)
)                                         
                                         

#### remove real jets (with high hadronic energy fraction) from SC collection
##### this improves the purity of the probe sample without affecting efficiency

process.JetsToRemoveFromSuperCluster = cms.EDFilter("CaloJetSelector",   
    src = cms.InputTag("ak5CaloJets"),
    cut = cms.string('pt>5 && energyFractionHadronic > 0.15')
)
process.goodSuperClustersClean = cms.EDProducer("CandViewCleaner",
    srcObject = cms.InputTag("goodSuperClusters"),
    module_label = cms.string(''),
    srcObjectsToRemove = cms.VInputTag(cms.InputTag("JetsToRemoveFromSuperCluster")),
    deltaRMin = cms.double(0.1)
)

#  Photons!!! ################ 
process.goodPhotons = cms.EDFilter(
    "PhotonSelector",
    src = cms.InputTag( PHOTON_COLL ),
    cut = cms.string(PHOTON_CUTS)
    )


process.sc_sequence = cms.Sequence(
    process.superClusters +
    process.superClusterCands +
    process.goodSuperClusters +
    process.JetsToRemoveFromSuperCluster +
    process.goodSuperClustersClean +
    process.goodPhotons
    )


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

process.GsfMatchedSuperClusterCands = cms.EDProducer("ElectronMatchedCandidateProducer",
   src     = cms.InputTag("goodSuperClustersClean"),
   ReferenceElectronCollection = cms.untracked.InputTag("goodElectrons"),
   deltaR =  cms.untracked.double(0.3)
)

process.GsfMatchedPhotonCands = process.GsfMatchedSuperClusterCands.clone()
process.GsfMatchedPhotonCands.src = cms.InputTag("goodPhotons")

            

##    _____ _           _                     ___    _ 
##   | ____| | ___  ___| |_ _ __ ___  _ __   |_ _|__| |
##   |  _| | |/ _ \/ __| __| '__/ _ \| '_ \   | |/ _` |
##   | |___| |  __/ (__| |_| | | (_) | | | |  | | (_| |
##   |_____|_|\___|\___|\__|_|  \___/|_| |_| |___\__,_|
##   
# Electron ID  ######
process.PassingWP95 = process.goodElectrons.clone()
process.PassingWP95.cut = cms.string(
    process.goodElectrons.cut.value() +
    " && (gsfTrack.hitPattern().numberOfLostHits(\'MISSING_INNER_HITS\') <= 1)"
    " && ((isEB"
    " && ( dr03TkSumPt/p4.Pt < 0.15 && dr03EcalRecHitSumEt/p4.Pt < 2.0 && dr03HcalTowerSumEt/p4.Pt < 0.12 )" 
    " && (sigmaIetaIeta<0.01)"
    " && ( -0.8<deltaPhiSuperClusterTrackAtVtx<0.8 )"
    " && ( -0.007<deltaEtaSuperClusterTrackAtVtx<0.007 )"
    " && (hadronicOverEm<0.15)"
    ")"
    " || (isEE"
    " && (dr03TkSumPt/p4.Pt < 0.08 && dr03EcalRecHitSumEt/p4.Pt < 0.06  && dr03HcalTowerSumEt/p4.Pt < 0.05 )"  
    " && (sigmaIetaIeta<0.03)"
    " && ( -0.7<deltaPhiSuperClusterTrackAtVtx<0.7 )"
    " && ( -0.01<deltaEtaSuperClusterTrackAtVtx<0.01 )"
    " && (hadronicOverEm<0.07) "
    "))"
    )
process.PassingWP90 = process.goodElectrons.clone()
process.PassingWP90.cut = cms.string(
    process.goodElectrons.cut.value() +
    " && (gsfTrack.hitPattern().numberOfLostHits(\'MISSING_INNER_HITS\') == 0 && !(-0.02<convDist<0.02 && -0.02<convDcot<0.02))"
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
process.PassingWP85 = process.goodElectrons.clone()
process.PassingWP85.cut = cms.string(
    process.goodElectrons.cut.value() +
    " && (gsfTrack.hitPattern().numberOfLostHits(\'MISSING_INNER_HITS\') == 0 && !(-0.02<convDist<0.02 && -0.02<convDcot<0.02))"
    " && ((isEB"
    " && ( dr03TkSumPt/p4.Pt <0.09 && dr03EcalRecHitSumEt/p4.Pt < 0.08 && dr03HcalTowerSumEt/p4.Pt  < 0.1 )"
    " && (sigmaIetaIeta<0.01)"
    " && ( -0.6<deltaPhiSuperClusterTrackAtVtx<0.6 )"
    " && ( -0.006<deltaEtaSuperClusterTrackAtVtx<0.006 )"
    " && (hadronicOverEm<0.04)"
    ")"
    " || (isEE"
    " && ( dr03TkSumPt/p4.Pt <0.05 && dr03EcalRecHitSumEt/p4.Pt < 0.05 && dr03HcalTowerSumEt/p4.Pt  < 0.025 )"
    " && (sigmaIetaIeta<0.03)"
    " && ( -0.04<deltaPhiSuperClusterTrackAtVtx<0.04 )"
    " && ( -0.007<deltaEtaSuperClusterTrackAtVtx<0.007 )"
    " && (hadronicOverEm<0.025) "
    "))"
    ) 
process.PassingWP80 = process.goodElectrons.clone()
process.PassingWP80.cut = cms.string(
    process.goodElectrons.cut.value() +
    " && (gsfTrack.hitPattern().numberOfLostHits(\'MISSING_INNER_HITS\') == 0 && !(-0.02<convDist<0.02 && -0.02<convDcot<0.02))"
    " && ((isEB"
    " && ( dr03TkSumPt/p4.Pt <0.09 && dr03EcalRecHitSumEt/p4.Pt < 0.07 && dr03HcalTowerSumEt/p4.Pt  < 0.1 )"
    " && (sigmaIetaIeta<0.01)"
    " && ( -0.06<deltaPhiSuperClusterTrackAtVtx<0.06 )"
    " && ( -0.004<deltaEtaSuperClusterTrackAtVtx<0.004 )"
    " && (hadronicOverEm<0.04)"
    ")"
    " || (isEE"
    " && ( dr03TkSumPt/p4.Pt <0.04 && dr03EcalRecHitSumEt/p4.Pt < 0.05 && dr03HcalTowerSumEt/p4.Pt  < 0.025 )"
    " && (sigmaIetaIeta<0.03)"
    " && ( -0.03<deltaPhiSuperClusterTrackAtVtx<0.03 )"
    " && ( -0.007<deltaEtaSuperClusterTrackAtVtx<0.007 )"
    " && (hadronicOverEm<0.025) "
    "))"
    ) 
process.PassingWP70 = process.goodElectrons.clone()
process.PassingWP70.cut = cms.string(
    process.goodElectrons.cut.value() +
    " && (gsfTrack.hitPattern().numberOfLostHits(\'MISSING_INNER_HITS\') == 0 && !(-0.02<convDist<0.02 && -0.02<convDcot<0.02))"
    " && ((isEB"
    " && ( dr03TkSumPt/p4.Pt <0.05 && dr03EcalRecHitSumEt/p4.Pt < 0.06 && dr03HcalTowerSumEt/p4.Pt  < 0.03 )"
    " && (sigmaIetaIeta<0.01)"
    " && ( -0.03<deltaPhiSuperClusterTrackAtVtx<0.03 )"
    " && ( -0.004<deltaEtaSuperClusterTrackAtVtx<0.004 )"
    " && (hadronicOverEm<0.025)"
    ")"
    " || (isEE"
    " && ( dr03TkSumPt/p4.Pt <0.025 && dr03EcalRecHitSumEt/p4.Pt < 0.025 && dr03HcalTowerSumEt/p4.Pt  < 0.02 )"
    " && (sigmaIetaIeta<0.03)"
    " && ( -0.02<deltaPhiSuperClusterTrackAtVtx<0.02 )"
    " && ( -0.005<deltaEtaSuperClusterTrackAtVtx<0.005 )"
    " && (hadronicOverEm<0.025) "
    "))"
    ) 
process.PassingWP60 = process.goodElectrons.clone()
process.PassingWP60.cut = cms.string(
    process.goodElectrons.cut.value() +
    " && (gsfTrack.hitPattern().numberOfLostHits(\'MISSING_INNER_HITS\') == 0 && !(-0.02<convDist<0.02 && -0.02<convDcot<0.02))"
    " && ((isEB"
    " && ( dr03TkSumPt/p4.Pt <0.04 && dr03EcalRecHitSumEt/p4.Pt < 0.04 && dr03HcalTowerSumEt/p4.Pt  < 0.03 )"
    " && (sigmaIetaIeta<0.01)"
    " && ( -0.025<deltaPhiSuperClusterTrackAtVtx<0.025 )"
    " && ( -0.004<deltaEtaSuperClusterTrackAtVtx<0.004 )"
    " && (hadronicOverEm<0.025)"
    ")"
    " || (isEE"
    " && ( dr03TkSumPt/p4.Pt <0.025 && dr03EcalRecHitSumEt/p4.Pt < 0.02 && dr03HcalTowerSumEt/p4.Pt  < 0.02 )"
    " && (sigmaIetaIeta<0.03)"
    " && ( -0.02<deltaPhiSuperClusterTrackAtVtx<0.02 )"
    " && ( -0.005<deltaEtaSuperClusterTrackAtVtx<0.005 )"
    " && (hadronicOverEm<0.025) "
    "))"
    ) 

##     ____ _  ____ 
##    / ___(_)/ ___|
##   | |   | | |    
##   | |___| | |___ 
##    \____|_|\____|
##   
process.load("RecoEgamma.ElectronIdentification.cutsInCategoriesElectronIdentificationV06_DataTuning_cfi")
process.load("RecoEgamma.ElectronIdentification.electronIdLikelihoodExt_cfi")

process.eIDSequence = cms.Sequence(
    process.eidVeryLoose+ 
    process.eidLoose+                
    process.eidMedium+
    process.eidTight+
    process.eidSuperTight+
    process.eidHyperTight1+
    process.eidHyperTight2+
    process.eidHyperTight3+
    process.eidHyperTight4+
    process.eidLikelihoodExt 
    )


# select a subset of the GsfElectron collection based on the quality stored in a ValueMap
process.PassingCicVeryLoose = cms.EDProducer("BtagGsfElectronSelector",
   input     = cms.InputTag( ELECTRON_COLL ),
   selection = cms.InputTag('eidVeryLoose'),
   cut       = cms.double(14.5) ### 15== passing all iso,id,tip cuts
)
process.PassingCicLoose = process.PassingCicVeryLoose.clone()
process.PassingCicLoose.selection = cms.InputTag('eidLoose')
process.PassingCicMedium = process.PassingCicVeryLoose.clone()
process.PassingCicMedium.selection = cms.InputTag('eidMedium')
process.PassingCicTight = process.PassingCicVeryLoose.clone()
process.PassingCicTight.selection = cms.InputTag('eidTight')
process.PassingCicSuperTight = process.PassingCicVeryLoose.clone()
process.PassingCicSuperTight.selection = cms.InputTag('eidSuperTight')
process.PassingCicHyperTight1 = process.PassingCicVeryLoose.clone()
process.PassingCicHyperTight1.selection = cms.InputTag('eidHyperTight1')
process.PassingCicHyperTight2 = process.PassingCicVeryLoose.clone()
process.PassingCicHyperTight2.selection = cms.InputTag('eidHyperTight2')
process.PassingCicHyperTight3 = process.PassingCicVeryLoose.clone()
process.PassingCicHyperTight3.selection = cms.InputTag('eidHyperTight3')
process.PassingCicHyperTight4 = process.PassingCicVeryLoose.clone()
process.PassingCicHyperTight4.selection = cms.InputTag('eidHyperTight4')


                         
##    _____     _                         __  __       _       _     _             
##   |_   _| __(_) __ _  __ _  ___ _ __  |  \/  | __ _| |_ ___| |__ (_)_ __   __ _ 
##     | || '__| |/ _` |/ _` |/ _ \ '__| | |\/| |/ _` | __/ __| '_ \| | '_ \ / _` |
##     | || |  | | (_| | (_| |  __/ |    | |  | | (_| | || (__| | | | | | | | (_| |
##     |_||_|  |_|\__, |\__, |\___|_|    |_|  |_|\__,_|\__\___|_| |_|_|_| |_|\__, |
##                |___/ |___/                                                |___/ 
##   
# Trigger  ##################
process.PassingHLT = cms.EDProducer("trgMatchedGsfElectronProducer",    
    InputProducer = cms.InputTag( ELECTRON_COLL ),                          
    hltTags = cms.VInputTag(cms.InputTag(HLTPath,"", HLTProcessName)),
    triggerEventTag = cms.untracked.InputTag("hltTriggerSummaryAOD","",HLTProcessName),
    triggerResultsTag = cms.untracked.InputTag("TriggerResults","",HLTProcessName)   
)

##    _____      _                        _  __     __             
##   | ____|_  _| |_ ___ _ __ _ __   __ _| | \ \   / /_ _ _ __ ___ 
##   |  _| \ \/ / __/ _ \ '__| '_ \ / _` | |  \ \ / / _` | '__/ __|
##   | |___ >  <| ||  __/ |  | | | | (_| | |   \ V / (_| | |  \__ \
##   |_____/_/\_\\__\___|_|  |_| |_|\__,_|_|    \_/ \__,_|_|  |___/
##   
## Here we show how to use a module to compute an external variable

process.superClusterDRToNearestJet = cms.EDProducer("DeltaRNearestJetComputer",
    probes = cms.InputTag("goodSuperClusters"),
       # ^^--- NOTA BENE: if probes are defined by ref, as in this case, 
       #       this must be the full collection, not the subset by refs.
    objects = cms.InputTag(JET_COLL),
    objectSelection = cms.string(JET_CUTS + " && pt > 20.0"),
)
process.JetMultiplicityInSCEvents = cms.EDProducer("CandMultiplicityCounter",
    probes = cms.InputTag("goodSuperClusters"),
    objects = cms.InputTag(JET_COLL),
    objectSelection = cms.string(JET_CUTS + " && pt > 20.0"),
)
process.SCConvRejVars = cms.EDProducer("ElectronConversionRejectionVars",
    probes = cms.InputTag("goodSuperClusters")
)
process.GsfConvRejVars = process.SCConvRejVars.clone()
process.GsfConvRejVars.probes = cms.InputTag( ELECTRON_COLL )
process.PhotonDRToNearestJet = process.superClusterDRToNearestJet.clone()
process.PhotonDRToNearestJet.probes =cms.InputTag("goodPhotons")
process.JetMultiplicityInPhotonEvents = process.JetMultiplicityInSCEvents.clone()
process.JetMultiplicityInPhotonEvents.probes = cms.InputTag("goodPhotons")
process.PhotonConvRejVars = process.SCConvRejVars.clone()
process.PhotonConvRejVars.probes = cms.InputTag("goodPhotons")

process.GsfDRToNearestJet = process.superClusterDRToNearestJet.clone()
process.GsfDRToNearestJet.probes = cms.InputTag( ELECTRON_COLL )
process.JetMultiplicityInGsfEvents = process.JetMultiplicityInSCEvents.clone()
process.JetMultiplicityInGsfEvents.probes = cms.InputTag( ELECTRON_COLL )

process.ext_ToNearestJet_sequence = cms.Sequence(
    #process.ak5PFResidual + 
    process.superClusterDRToNearestJet +
    process.JetMultiplicityInSCEvents +
    process.SCConvRejVars +
    process.PhotonDRToNearestJet +
    process.JetMultiplicityInPhotonEvents +    
    process.PhotonConvRejVars + 
    process.GsfDRToNearestJet +
    process.JetMultiplicityInGsfEvents +
    process.GsfConvRejVars
    )


##    _____             ____        __ _       _ _   _             
##   |_   _|_ _  __ _  |  _ \  ___ / _(_)_ __ (_) |_(_) ___  _ __  
##     | |/ _` |/ _` | | | | |/ _ \ |_| | '_ \| | __| |/ _ \| '_ \ 
##     | | (_| | (_| | | |_| |  __/  _| | | | | | |_| | (_) | | | |
##     |_|\__,_|\__, | |____/ \___|_| |_|_| |_|_|\__|_|\___/|_| |_|
##              |___/
## 
process.Tag = process.PassingHLT.clone()
process.Tag.InputProducer = cms.InputTag( "PassingWP80" )
process.TagMatchedSuperClusterCandsClean = cms.EDProducer("ElectronMatchedCandidateProducer",
   src     = cms.InputTag("goodSuperClustersClean"),
   ReferenceElectronCollection = cms.untracked.InputTag("Tag"),
   deltaR =  cms.untracked.double(0.3)
)
process.TagMatchedPhotonCands = process.TagMatchedSuperClusterCandsClean.clone()
process.TagMatchedPhotonCands.src     = cms.InputTag("goodPhotons")
process.WP95MatchedSuperClusterCandsClean = process.TagMatchedSuperClusterCandsClean.clone()
process.WP95MatchedSuperClusterCandsClean.ReferenceElectronCollection = cms.untracked.InputTag("PassingWP95")
process.WP90MatchedSuperClusterCandsClean = process.TagMatchedSuperClusterCandsClean.clone()
process.WP90MatchedSuperClusterCandsClean.ReferenceElectronCollection = cms.untracked.InputTag("PassingWP90")
process.WP85MatchedSuperClusterCandsClean = process.TagMatchedSuperClusterCandsClean.clone()
process.WP85MatchedSuperClusterCandsClean.ReferenceElectronCollection = cms.untracked.InputTag("PassingWP85")
process.WP80MatchedSuperClusterCandsClean = process.TagMatchedSuperClusterCandsClean.clone()
process.WP80MatchedSuperClusterCandsClean.ReferenceElectronCollection = cms.untracked.InputTag("PassingWP80")
process.WP70MatchedSuperClusterCandsClean = process.TagMatchedSuperClusterCandsClean.clone()
process.WP70MatchedSuperClusterCandsClean.ReferenceElectronCollection = cms.untracked.InputTag("PassingWP70")
process.WP60MatchedSuperClusterCandsClean = process.TagMatchedSuperClusterCandsClean.clone()
process.WP60MatchedSuperClusterCandsClean.ReferenceElectronCollection = cms.untracked.InputTag("PassingWP60")
process.CicVeryLooseMatchedSuperClusterCandsClean = process.TagMatchedSuperClusterCandsClean.clone()
process.CicVeryLooseMatchedSuperClusterCandsClean.ReferenceElectronCollection = cms.untracked.InputTag("PassingCicVeryLoose")
process.CicLooseMatchedSuperClusterCandsClean = process.TagMatchedSuperClusterCandsClean.clone()
process.CicLooseMatchedSuperClusterCandsClean.ReferenceElectronCollection = cms.untracked.InputTag("PassingCicLoose")
process.CicMediumMatchedSuperClusterCandsClean = process.TagMatchedSuperClusterCandsClean.clone()
process.CicMediumMatchedSuperClusterCandsClean.ReferenceElectronCollection = cms.untracked.InputTag("PassingCicMedium")
process.CicTightMatchedSuperClusterCandsClean = process.TagMatchedSuperClusterCandsClean.clone()
process.CicTightMatchedSuperClusterCandsClean.ReferenceElectronCollection = cms.untracked.InputTag("PassingCicTight")
process.CicSuperTightMatchedSuperClusterCandsClean = process.TagMatchedSuperClusterCandsClean.clone()
process.CicSuperTightMatchedSuperClusterCandsClean.ReferenceElectronCollection = cms.untracked.InputTag("PassingCicSuperTight")
process.CicHyperTight1MatchedSuperClusterCandsClean = process.TagMatchedSuperClusterCandsClean.clone()
process.CicHyperTight1MatchedSuperClusterCandsClean.ReferenceElectronCollection = cms.untracked.InputTag("PassingCicHyperTight1")
process.CicHyperTight2MatchedSuperClusterCandsClean = process.TagMatchedSuperClusterCandsClean.clone()
process.CicHyperTight2MatchedSuperClusterCandsClean.ReferenceElectronCollection = cms.untracked.InputTag("PassingCicHyperTight2")
process.CicHyperTight3MatchedSuperClusterCandsClean = process.TagMatchedSuperClusterCandsClean.clone()
process.CicHyperTight3MatchedSuperClusterCandsClean.ReferenceElectronCollection = cms.untracked.InputTag("PassingCicHyperTight3")
process.CicHyperTight4MatchedSuperClusterCandsClean = process.TagMatchedSuperClusterCandsClean.clone()
process.CicHyperTight4MatchedSuperClusterCandsClean.ReferenceElectronCollection = cms.untracked.InputTag("PassingCicHyperTight4")


process.WP95MatchedPhotonCands = process.GsfMatchedPhotonCands.clone()
process.WP95MatchedPhotonCands.ReferenceElectronCollection = cms.untracked.InputTag("PassingWP95")
process.WP90MatchedPhotonCands = process.GsfMatchedPhotonCands.clone()
process.WP90MatchedPhotonCands.ReferenceElectronCollection = cms.untracked.InputTag("PassingWP90")
process.WP85MatchedPhotonCands = process.GsfMatchedPhotonCands.clone()
process.WP85MatchedPhotonCands.ReferenceElectronCollection = cms.untracked.InputTag("PassingWP85")
process.WP80MatchedPhotonCands = process.GsfMatchedPhotonCands.clone()
process.WP80MatchedPhotonCands.ReferenceElectronCollection = cms.untracked.InputTag("PassingWP80")
process.WP70MatchedPhotonCands = process.GsfMatchedPhotonCands.clone()
process.WP70MatchedPhotonCands.ReferenceElectronCollection = cms.untracked.InputTag("PassingWP70")
process.WP60MatchedPhotonCands = process.GsfMatchedPhotonCands.clone()
process.WP60MatchedPhotonCands.ReferenceElectronCollection = cms.untracked.InputTag("PassingWP60")
process.CicVeryLooseMatchedPhotonCands = process.GsfMatchedPhotonCands.clone()
process.CicVeryLooseMatchedPhotonCands.ReferenceElectronCollection = cms.untracked.InputTag("PassingCicVeryLoose")
process.CicLooseMatchedPhotonCands = process.GsfMatchedPhotonCands.clone()
process.CicLooseMatchedPhotonCands.ReferenceElectronCollection = cms.untracked.InputTag("PassingCicLoose")
process.CicMediumMatchedPhotonCands = process.GsfMatchedPhotonCands.clone()
process.CicMediumMatchedPhotonCands.ReferenceElectronCollection = cms.untracked.InputTag("PassingCicMedium")
process.CicTightMatchedPhotonCands = process.GsfMatchedPhotonCands.clone()
process.CicTightMatchedPhotonCands.ReferenceElectronCollection = cms.untracked.InputTag("PassingCicTight")
process.CicSuperTightMatchedPhotonCands = process.GsfMatchedPhotonCands.clone()
process.CicSuperTightMatchedPhotonCands.ReferenceElectronCollection = cms.untracked.InputTag("PassingCicSuperTight")
process.CicHyperTight1MatchedPhotonCands = process.GsfMatchedPhotonCands.clone()
process.CicHyperTight1MatchedPhotonCands.ReferenceElectronCollection = cms.untracked.InputTag("PassingCicHyperTight1")
process.CicHyperTight2MatchedPhotonCands = process.GsfMatchedPhotonCands.clone()
process.CicHyperTight2MatchedPhotonCands.ReferenceElectronCollection = cms.untracked.InputTag("PassingCicHyperTight2")
process.CicHyperTight3MatchedPhotonCands = process.GsfMatchedPhotonCands.clone()
process.CicHyperTight3MatchedPhotonCands.ReferenceElectronCollection = cms.untracked.InputTag("PassingCicHyperTight3")
process.CicHyperTight4MatchedPhotonCands = process.GsfMatchedPhotonCands.clone()
process.CicHyperTight4MatchedPhotonCands.ReferenceElectronCollection = cms.untracked.InputTag("PassingCicHyperTight4")

process.ele_sequence = cms.Sequence(
    process.goodElectrons +
    process.GsfMatchedSuperClusterCands +
    process.GsfMatchedPhotonCands +
    process.PassingWP95 +
    process.PassingWP90 +
    process.PassingWP85 +
    process.PassingWP80 +
    process.PassingWP70 +
    process.PassingWP60 +
    process.PassingCicVeryLoose +
    process.PassingCicLoose +
    process.PassingCicMedium +
    process.PassingCicTight +
    process.PassingCicSuperTight +
    process.PassingCicHyperTight1 +
    process.PassingCicHyperTight2 +
    process.PassingCicHyperTight3 +
    process.PassingCicHyperTight4 +       
    process.PassingHLT +
    process.Tag +
    process.TagMatchedSuperClusterCandsClean +
    process.TagMatchedPhotonCands +
    process.WP95MatchedSuperClusterCandsClean +
    process.WP90MatchedSuperClusterCandsClean +
    process.WP85MatchedSuperClusterCandsClean +
    process.WP80MatchedSuperClusterCandsClean +
    process.WP70MatchedSuperClusterCandsClean +
    process.WP60MatchedSuperClusterCandsClean +    
    process.CicVeryLooseMatchedSuperClusterCandsClean +
    process.CicLooseMatchedSuperClusterCandsClean +
    process.CicMediumMatchedSuperClusterCandsClean +
    process.CicTightMatchedSuperClusterCandsClean +
    process.CicSuperTightMatchedSuperClusterCandsClean +
    process.CicHyperTight1MatchedSuperClusterCandsClean +
    process.CicHyperTight2MatchedSuperClusterCandsClean +
    process.CicHyperTight3MatchedSuperClusterCandsClean +
    process.CicHyperTight4MatchedSuperClusterCandsClean +
    process.WP95MatchedPhotonCands +
    process.WP90MatchedPhotonCands +
    process.WP85MatchedPhotonCands +
    process.WP80MatchedPhotonCands +
    process.WP70MatchedPhotonCands +
    process.WP60MatchedPhotonCands +    
    process.CicVeryLooseMatchedPhotonCands +
    process.CicLooseMatchedPhotonCands +
    process.CicMediumMatchedPhotonCands +
    process.CicTightMatchedPhotonCands +
    process.CicSuperTightMatchedPhotonCands +
    process.CicHyperTight1MatchedPhotonCands +
    process.CicHyperTight2MatchedPhotonCands +
    process.CicHyperTight3MatchedPhotonCands +
    process.CicHyperTight4MatchedPhotonCands         
    )


##    _____ ___   ____    ____       _          
##   |_   _( _ ) |  _ \  |  _ \ __ _(_)_ __ ___ 
##     | | / _ \/\ |_) | | |_) / _` | | '__/ __|
##     | || (_>  <  __/  |  __/ (_| | | |  \__ \
##     |_| \___/\/_|     |_|   \__,_|_|_|  |___/
##                                              
##   
#  Tag & probe selection ######
process.tagSC = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("Tag goodSuperClustersClean"), # charge coniugate states are implied
    checkCharge = cms.bool(False),                           
    cut   = cms.string("40 < mass < 1000"),
)

process.tagPhoton = process.tagSC.clone()
process.tagPhoton.decay = cms.string("Tag goodPhotons")
process.GsfGsf = process.tagSC.clone()
process.GsfGsf.decay = cms.string("goodElectrons goodElectrons")
process.tagGsf = process.tagSC.clone()
process.tagGsf.decay = cms.string("Tag goodElectrons")
process.tagWP95 = process.tagSC.clone()
process.tagWP95.decay = cms.string("Tag PassingWP95")
process.tagWP90 = process.tagSC.clone()
process.tagWP90.decay = cms.string("Tag PassingWP90")
process.tagWP85 = process.tagSC.clone()
process.tagWP85.decay = cms.string("Tag PassingWP85")
process.tagWP80 = process.tagSC.clone()
process.tagWP80.decay = cms.string("Tag PassingWP80")
process.tagWP70 = process.tagSC.clone()
process.tagWP70.decay = cms.string("Tag PassingWP70")
process.tagWP60 = process.tagSC.clone()
process.tagWP60.decay = cms.string("Tag PassingWP60")
process.tagCicVeryLoose = process.tagSC.clone()
process.tagCicVeryLoose.decay = cms.string("Tag PassingCicVeryLoose")
process.tagCicLoose = process.tagSC.clone()
process.tagCicLoose.decay = cms.string("Tag PassingCicLoose")
process.tagCicMedium = process.tagSC.clone()
process.tagCicMedium.decay = cms.string("Tag PassingCicMedium")
process.tagCicTight = process.tagSC.clone()
process.tagCicTight.decay = cms.string("Tag PassingCicTight")
process.tagCicSuperTight = process.tagSC.clone()
process.tagCicSuperTight.decay = cms.string("Tag PassingCicSuperTight")
process.tagCicHyperTight1 = process.tagSC.clone()
process.tagCicHyperTight1.decay = cms.string("Tag PassingCicHyperTight1")
process.tagCicHyperTight2 = process.tagSC.clone()
process.tagCicHyperTight2.decay = cms.string("Tag PassingCicHyperTight2")
process.tagCicHyperTight3 = process.tagSC.clone()
process.tagCicHyperTight3.decay = cms.string("Tag PassingCicHyperTight3")
process.tagCicHyperTight4 = process.tagSC.clone()
process.tagCicHyperTight4.decay = cms.string("Tag PassingCicHyperTight4")
process.elecMet = process.tagSC.clone()
process.elecMet.decay = cms.string("pfMet PassingWP90")
process.elecMet.cut = cms.string("mt > 0")

process.CSVarsTagGsf = cms.EDProducer("ColinsSoperVariablesComputer",
    parentBoson = cms.InputTag("tagGsf")
)
process.CSVarsGsfGsf = process.CSVarsTagGsf.clone()
process.CSVarsGsfGsf.parentBoson = cms.InputTag("GsfGsf")



process.allTagsAndProbes = cms.Sequence(
    process.tagSC +
    process.tagPhoton +
    process.tagGsf +
    process.GsfGsf +
    process.tagWP95 +
    process.tagWP90 +
    process.tagWP85 +
    process.tagWP80 +
    process.tagWP70 +
    process.tagWP60 +
    process.tagCicVeryLoose +
    process.tagCicLoose +
    process.tagCicMedium +
    process.tagCicTight +
    process.tagCicSuperTight +
    process.tagCicHyperTight1 +
    process.tagCicHyperTight2 +
    process.tagCicHyperTight3 +
    process.tagCicHyperTight4 +
    process.elecMet + 
    process.CSVarsTagGsf +
    process.CSVarsGsfGsf
)

##    __  __  ____   __  __       _       _               
##   |  \/  |/ ___| |  \/  | __ _| |_ ___| |__   ___  ___ 
##   | |\/| | |     | |\/| |/ _` | __/ __| '_ \ / _ \/ __|
##   | |  | | |___  | |  | | (_| | || (__| | | |  __/\__ \
##   |_|  |_|\____| |_|  |_|\__,_|\__\___|_| |_|\___||___/
##                                                        
process.McMatchTag = cms.EDProducer("MCTruthDeltaRMatcherNew",
    matchPDGId = cms.vint32(11),
    src = cms.InputTag("Tag"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("genParticles"),
    checkCharge = cms.bool(True)
)
process.McMatchSC = cms.EDProducer("MCTruthDeltaRMatcherNew",
    matchPDGId = cms.vint32(11),
    src = cms.InputTag("goodSuperClustersClean"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("genParticles")
)
process.McMatchPhoton = process.McMatchSC.clone()
process.McMatchPhoton.src = cms.InputTag("goodPhotons")
process.McMatchGsf = process.McMatchTag.clone()
process.McMatchGsf.src = cms.InputTag("goodElectrons")
process.McMatchWP95 = process.McMatchTag.clone()
process.McMatchWP95.src = cms.InputTag("PassingWP95")
process.McMatchWP90 = process.McMatchTag.clone()
process.McMatchWP90.src = cms.InputTag("PassingWP90")
process.McMatchWP85 = process.McMatchTag.clone()
process.McMatchWP85.src = cms.InputTag("PassingWP85")
process.McMatchWP80 = process.McMatchTag.clone()
process.McMatchWP80.src = cms.InputTag("PassingWP80")
process.McMatchWP70 = process.McMatchTag.clone()
process.McMatchWP70.src = cms.InputTag("PassingWP70")
process.McMatchWP60 = process.McMatchTag.clone()
process.McMatchWP60.src = cms.InputTag("PassingWP60")
process.McMatchCicVeryLoose = process.McMatchTag.clone()
process.McMatchCicVeryLoose.src = cms.InputTag("PassingCicVeryLoose")
process.McMatchCicLoose = process.McMatchTag.clone()
process.McMatchCicLoose.src = cms.InputTag("PassingCicLoose")
process.McMatchCicMedium = process.McMatchTag.clone()
process.McMatchCicMedium.src = cms.InputTag("PassingCicMedium")
process.McMatchCicTight = process.McMatchTag.clone()
process.McMatchCicTight.src = cms.InputTag("PassingCicTight")
process.McMatchCicSuperTight = process.McMatchTag.clone()
process.McMatchCicSuperTight.src = cms.InputTag("PassingCicSuperTight")
process.McMatchCicHyperTight1 = process.McMatchTag.clone()
process.McMatchCicHyperTight1.src = cms.InputTag("PassingCicHyperTight1")
process.McMatchCicHyperTight2 = process.McMatchTag.clone()
process.McMatchCicHyperTight2.src = cms.InputTag("PassingCicHyperTight2")
process.McMatchCicHyperTight3 = process.McMatchTag.clone()
process.McMatchCicHyperTight3.src = cms.InputTag("PassingCicHyperTight3")
process.McMatchCicHyperTight4 = process.McMatchTag.clone()
process.McMatchCicHyperTight4.src = cms.InputTag("PassingCicHyperTight4")
    
process.mc_sequence = cms.Sequence(
   process.McMatchTag +
   process.McMatchSC +
   process.McMatchPhoton +
   process.McMatchGsf + 
   process.McMatchWP95 +
   process.McMatchWP90 +
   process.McMatchWP85 +
   process.McMatchWP80 +
   process.McMatchWP70 +   
   process.McMatchWP60 +
   process.McMatchCicVeryLoose +
   process.McMatchCicLoose +
   process.McMatchCicMedium +
   process.McMatchCicTight +
   process.McMatchCicSuperTight +
   process.McMatchCicHyperTight1 +
   process.McMatchCicHyperTight2 +
   process.McMatchCicHyperTight3 +
   process.McMatchCicHyperTight4    
)

############################################################################
##    _____           _       _ ____            _            _   _  ____  ##
##   |_   _|_ _  __ _( )_ __ ( )  _ \ _ __ ___ | |__   ___  | \ | |/ ___| ##
##     | |/ _` |/ _` |/| '_ \|/| |_) | '__/ _ \| '_ \ / _ \ |  \| | |  _  ##
##     | | (_| | (_| | | | | | |  __/| | | (_) | |_) |  __/ | |\  | |_| | ##
##     |_|\__,_|\__, | |_| |_| |_|   |_|  \___/|_.__/ \___| |_| \_|\____| ##
##              |___/                                                     ##
##                                                                        ##
############################################################################
##    ____                      _     _           
##   |  _ \ ___ _   _ ___  __ _| |__ | | ___  ___ 
##   | |_) / _ \ | | / __|/ _` | '_ \| |/ _ \/ __|
##   |  _ <  __/ |_| \__ \ (_| | |_) | |  __/\__ \
##   |_| \_\___|\__,_|___/\__,_|_.__/|_|\___||___/
##
## I define some common variables for re-use later.
## This will save us repeating the same code for each efficiency category
ZVariablesToStore = cms.PSet(
    eta = cms.string("eta"),
    pt  = cms.string("pt"),
    phi  = cms.string("phi"),
    et  = cms.string("et"),
    e  = cms.string("energy"),
    p  = cms.string("p"),
    px  = cms.string("px"),
    py  = cms.string("py"),
    pz  = cms.string("pz"),
    theta  = cms.string("theta"),    
    vx     = cms.string("vx"),
    vy     = cms.string("vy"),
    vz     = cms.string("vz"),
    rapidity  = cms.string("rapidity"),
    mass  = cms.string("mass"),
    mt  = cms.string("mt"),    
)   

ProbeVariablesToStore = cms.PSet(
    probe_gsfEle_eta = cms.string("eta"),
    probe_gsfEle_pt  = cms.string("pt"),
    probe_gsfEle_phi  = cms.string("phi"),
    probe_gsfEle_et  = cms.string("et"),
    probe_gsfEle_e  = cms.string("energy"),
    probe_gsfEle_p  = cms.string("p"),
    probe_gsfEle_px  = cms.string("px"),
    probe_gsfEle_py  = cms.string("py"),
    probe_gsfEle_pz  = cms.string("pz"),
    probe_gsfEle_theta  = cms.string("theta"),    
    probe_gsfEle_charge = cms.string("charge"),
    probe_gsfEle_rapidity  = cms.string("rapidity"),
    probe_gsfEle_missingHits = cms.string("gsfTrack.hitPattern().numberOfLostHits(\'MISSING_INNER_HITS\')"),
    probe_gsfEle_convDist = cms.string("convDist"),
    probe_gsfEle_convDcot = cms.string("convDcot"),
    probe_gsfEle_convRadius = cms.string("convRadius"),        
    probe_gsfEle_hasValidHitInFirstPixelBarrel = cms.string("gsfTrack.hitPattern().hasValidHitInFirstPixelBarrel()"),
    ## super cluster quantities
    probe_sc_energy = cms.string("superCluster.energy"),
    probe_sc_et    = cms.string("superCluster.energy*sin(superClusterPosition.theta)"),    
    probe_sc_x      = cms.string("superCluster.x"),
    probe_sc_y      = cms.string("superCluster.y"),
    probe_sc_z      = cms.string("superCluster.z"),
    probe_sc_eta    = cms.string("superCluster.eta"),
    probe_sc_theta  = cms.string("superClusterPosition.theta"),   
    probe_sc_phi    = cms.string("superCluster.phi"),
    probe_sc_size   = cms.string("superCluster.size"), # number of hits
    ## track quantities
    probe_track_p      = cms.string("gsfTrack.p"),
    probe_track_pt     = cms.string("gsfTrack.pt"),    
    probe_track_px     = cms.string("gsfTrack.px"),
    probe_track_py     = cms.string("gsfTrack.py"),
    probe_track_pz     = cms.string("gsfTrack.pz"),
    probe_track_eta    = cms.string("gsfTrack.eta"),
    probe_track_theta  = cms.string("gsfTrack.theta"),   
    probe_track_phi    = cms.string("gsfTrack.phi"),
    probe_track_vx     = cms.string("gsfTrack.vx"),
    probe_track_vy     = cms.string("gsfTrack.vy"),
    probe_track_vz     = cms.string("gsfTrack.vz"),    
    probe_track_dxy    = cms.string("gsfTrack.dxy"),
    probe_track_d0     = cms.string("gsfTrack.d0"),
    probe_track_dsz    = cms.string("gsfTrack.dsz"),
    probe_track_charge = cms.string("gsfTrack.charge"),
    probe_track_qoverp = cms.string("gsfTrack.qoverp"),
    probe_track_normalizedChi2 = cms.string("gsfTrack.normalizedChi2"),
    ## isolation 
    probe_gsfEle_trackiso = cms.string("dr03TkSumPt"),
    probe_gsfEle_ecaliso  = cms.string("dr03EcalRecHitSumEt"),
    probe_gsfEle_hcaliso  = cms.string("dr03HcalTowerSumEt"),
    ## classification, location, etc.    
    probe_gsfEle_classification = cms.string("classification"),
    probe_gsfEle_numberOfBrems  = cms.string("numberOfBrems"),     
    probe_gsfEle_bremFraction   = cms.string("fbrem"),
    probe_gsfEle_mva            = cms.string("mva"),        
    probe_gsfEle_deltaEta       = cms.string("deltaEtaSuperClusterTrackAtVtx"),
    probe_gsfEle_deltaPhi       = cms.string("deltaPhiSuperClusterTrackAtVtx"),
    probe_gsfEle_deltaPhiOut    = cms.string("deltaPhiSeedClusterTrackAtCalo"),
    probe_gsfEle_deltaEtaOut    = cms.string("deltaEtaSeedClusterTrackAtCalo"),
    probe_gsfEle_isEB           = cms.string("isEB"),
    probe_gsfEle_isEE           = cms.string("isEE"),
    probe_gsfEle_isGap          = cms.string("isGap"),
    ## Hcal energy over Ecal Energy
    probe_gsfEle_HoverE         = cms.string("hcalOverEcal"),    
    probe_gsfEle_EoverP         = cms.string("eSuperClusterOverP"),
    probe_gsfEle_eSeedClusterOverP = cms.string("eSeedClusterOverP"),    
    ## Cluster shape information
    probe_gsfEle_sigmaEtaEta  = cms.string("sigmaEtaEta"),
    probe_gsfEle_sigmaIetaIeta = cms.string("sigmaIetaIeta"),
    probe_gsfEle_e1x5               = cms.string("e1x5"),
    probe_gsfEle_e2x5Max            = cms.string("e2x5Max"),
    probe_gsfEle_e5x5               = cms.string("e5x5"),
    ## is ECAL driven ? is Track driven ?
    probe_gsfEle_ecalDrivenSeed     = cms.string("ecalDrivenSeed"),
    probe_gsfEle_trackerDrivenSeed  = cms.string("trackerDrivenSeed")
)


TagVariablesToStore = cms.PSet(
    gsfEle_eta = cms.string("eta"),
    gsfEle_pt  = cms.string("pt"),
    gsfEle_phi  = cms.string("phi"),
    gsfEle_et  = cms.string("et"),
    gsfEle_e  = cms.string("energy"),
    gsfEle_p  = cms.string("p"),
    gsfEle_px  = cms.string("px"),
    gsfEle_py  = cms.string("py"),
    gsfEle_pz  = cms.string("pz"),
    gsfEle_theta  = cms.string("theta"),    
    gsfEle_charge = cms.string("charge"),
    gsfEle_rapidity  = cms.string("rapidity"),
    gsfEle_missingHits = cms.string("gsfTrack.hitPattern().numberOfLostHits(\'MISSING_INNER_HITS\')"),
    gsfEle_convDist = cms.string("convDist"),
    gsfEle_convDcot = cms.string("convDcot"),
    gsfEle_convRadius = cms.string("convRadius"),     
    gsfEle_hasValidHitInFirstPixelBarrel = cms.string("gsfTrack.hitPattern().hasValidHitInFirstPixelBarrel()"),
    ## super cluster quantities
    sc_energy = cms.string("superCluster.energy"),
    sc_et     = cms.string("superCluster.energy*sin(superClusterPosition.theta)"),    
    sc_x      = cms.string("superCluster.x"),
    sc_y      = cms.string("superCluster.y"),
    sc_z      = cms.string("superCluster.z"),
    sc_eta    = cms.string("superCluster.eta"),
    sc_theta  = cms.string("superClusterPosition.theta"),      
    sc_phi    = cms.string("superCluster.phi"),
    sc_size   = cms.string("superCluster.size"), # number of hits
    ## track quantities
    track_p      = cms.string("gsfTrack.p"),
    track_pt     = cms.string("gsfTrack.pt"),    
    track_px     = cms.string("gsfTrack.px"),
    track_py     = cms.string("gsfTrack.py"),
    track_pz     = cms.string("gsfTrack.pz"),
    track_eta    = cms.string("gsfTrack.eta"),
    track_theta  = cms.string("gsfTrack.theta"),   
    track_phi    = cms.string("gsfTrack.phi"),
    track_vx     = cms.string("gsfTrack.vx"),
    track_vy     = cms.string("gsfTrack.vy"),
    track_vz     = cms.string("gsfTrack.vz"),    
    track_dxy    = cms.string("gsfTrack.dxy"),
    track_d0     = cms.string("gsfTrack.d0"),
    track_dsz    = cms.string("gsfTrack.dsz"),
    track_charge = cms.string("gsfTrack.charge"),
    track_qoverp = cms.string("gsfTrack.qoverp"),
    track_normalizedChi2 = cms.string("gsfTrack.normalizedChi2"),    
    ## isolation 
    gsfEle_trackiso = cms.string("dr03TkSumPt"),
    gsfEle_ecaliso  = cms.string("dr03EcalRecHitSumEt"),
    gsfEle_hcaliso  = cms.string("dr03HcalTowerSumEt"),
    ## classification, location, etc.    
    gsfEle_classification = cms.string("classification"),
    gsfEle_numberOfBrems  = cms.string("numberOfBrems"),     
    gsfEle_bremFraction   = cms.string("fbrem"),
    gsfEle_mva            = cms.string("mva"),        
    gsfEle_deltaEta       = cms.string("deltaEtaSuperClusterTrackAtVtx"),
    gsfEle_deltaPhi       = cms.string("deltaPhiSuperClusterTrackAtVtx"),
    gsfEle_deltaPhiOut    = cms.string("deltaPhiSeedClusterTrackAtCalo"),
    gsfEle_deltaEtaOut    = cms.string("deltaEtaSeedClusterTrackAtCalo"),
    gsfEle_isEB           = cms.string("isEB"),
    gsfEle_isEE           = cms.string("isEE"),
    gsfEle_isGap          = cms.string("isGap"),
    ## Hcal energy over Ecal Energy
    gsfEle_HoverE         = cms.string("hcalOverEcal"),    
    gsfEle_EoverP         = cms.string("eSuperClusterOverP"),
    gsfEle_eSeedClusterOverP = cms.string("eSeedClusterOverP"),  
    ## Cluster shape information
    gsfEle_sigmaEtaEta  = cms.string("sigmaEtaEta"),
    gsfEle_sigmaIetaIeta = cms.string("sigmaIetaIeta"),
    gsfEle_e1x5               = cms.string("e1x5"),
    gsfEle_e2x5Max            = cms.string("e2x5Max"),
    gsfEle_e5x5               = cms.string("e5x5"),
    ## is ECAL driven ? is Track driven ?
    gsfEle_ecalDrivenSeed     = cms.string("ecalDrivenSeed"),
    gsfEle_trackerDrivenSeed  = cms.string("trackerDrivenSeed")
)

CommonStuffForGsfElectronProbe = cms.PSet(
    variables = cms.PSet(ProbeVariablesToStore),
    addRunLumiInfo   =  cms.bool (True),
    addEventVariablesInfo   =  cms.bool (True),
    pairVariables =  cms.PSet(ZVariablesToStore),
    pairFlags     =  cms.PSet(
          mass60to120 = cms.string("60 < mass < 120")
    ),
    tagVariables   =  cms.PSet(TagVariablesToStore),
    tagFlags     =  cms.PSet(
          passingGsf = cms.InputTag("goodElectrons"),
          isWP95 = cms.InputTag("PassingWP95"),
          isWP90 = cms.InputTag("PassingWP90"),
          isWP85 = cms.InputTag("PassingWP85"),          
          isWP80 = cms.InputTag("PassingWP80"),
          isWP70 = cms.InputTag("PassingWP70"),
          isWP60 = cms.InputTag("PassingWP60"),
          isCicVeryLoose = cms.InputTag("PassingCicVeryLoose"),
          isCicLoose = cms.InputTag("PassingCicLoose"),
          isCicMedium = cms.InputTag("PassingCicMedium"),
          isCicTight = cms.InputTag("PassingCicTight"),
          isCicSuperTight = cms.InputTag("PassingCicSuperTight"),          
          isCicHyperTight1 = cms.InputTag("PassingCicHyperTight1"),
          isCicHyperTight2 = cms.InputTag("PassingCicHyperTight2"),
          isCicHyperTight3 = cms.InputTag("PassingCicHyperTight3"),
          isCicHyperTight4 = cms.InputTag("PassingCicHyperTight4"),          
          passingHLT = cms.InputTag("PassingHLT")     
    ),    
)

CommonStuffForSuperClusterProbe = CommonStuffForGsfElectronProbe.clone()
CommonStuffForSuperClusterProbe.variables = cms.PSet(
    probe_eta = cms.string("eta"),
    probe_pt  = cms.string("pt"),
    probe_phi  = cms.string("phi"),
    probe_et  = cms.string("et"),
    probe_e  = cms.string("energy"),
    probe_p  = cms.string("p"),
    probe_px  = cms.string("px"),
    probe_py  = cms.string("py"),
    probe_pz  = cms.string("pz"),
    probe_theta  = cms.string("theta"),
    )


if MC_flag:
    mcTruthCommonStuff = cms.PSet(
        isMC = cms.bool(MC_flag),
        tagMatches = cms.InputTag("McMatchTag"),
        motherPdgId = cms.vint32(22,23),
        makeMCUnbiasTree = cms.bool(MC_flag),
        checkMotherInUnbiasEff = cms.bool(MC_flag),
        mcVariables = cms.PSet(
        probe_eta = cms.string("eta"),
        probe_pt  = cms.string("pt"),
        probe_phi  = cms.string("phi"),
        probe_et  = cms.string("et"),
        probe_e  = cms.string("energy"),
        probe_p  = cms.string("p"),
        probe_px  = cms.string("px"),
        probe_py  = cms.string("py"),
        probe_pz  = cms.string("pz"),
        probe_theta  = cms.string("theta"),    
        probe_vx     = cms.string("vx"),
        probe_vy     = cms.string("vy"),
        probe_vz     = cms.string("vz"),   
        probe_charge = cms.string("charge"),
        probe_rapidity  = cms.string("rapidity"),    
        probe_mass  = cms.string("mass"),
        probe_mt  = cms.string("mt"),    
        ),
        mcFlags     =  cms.PSet(
        probe_flag = cms.string("pt>0")
        ),      
        )
else:
     mcTruthCommonStuff = cms.PSet(
         isMC = cms.bool(False)
         )


##    ____   ____       __     ____      __ 
##   / ___| / ___|      \ \   / ___|___ / _|
##   \___ \| |      _____\ \ | |  _/ __| |_ 
##    ___) | |___  |_____/ / | |_| \__ \  _|
##   |____/ \____|      /_/   \____|___/_|  
##
## super cluster --> gsf electron
process.SuperClusterToGsfElectron = cms.EDAnalyzer("TagProbeFitTreeProducer",
    ## pick the defaults
    CommonStuffForSuperClusterProbe, mcTruthCommonStuff,
    # choice of tag and probe pairs, and arbitration                 
    tagProbePairs = cms.InputTag("tagSC"),
    arbitration   = cms.string("Random2"),                      
    flags = cms.PSet(
        probe_passingGsf = cms.InputTag("GsfMatchedSuperClusterCands"),        
        probe_isWP95 = cms.InputTag("WP95MatchedSuperClusterCandsClean"),
        probe_isWP90 = cms.InputTag("WP90MatchedSuperClusterCandsClean"),
        probe_isWP85 = cms.InputTag("WP85MatchedSuperClusterCandsClean"),        
        probe_isWP80 = cms.InputTag("WP80MatchedSuperClusterCandsClean"),
        probe_isWP70 = cms.InputTag("WP70MatchedSuperClusterCandsClean"),
        probe_isWP60 = cms.InputTag("WP60MatchedSuperClusterCandsClean"),
        probe_isCicVeryLoose = cms.InputTag("CicVeryLooseMatchedSuperClusterCandsClean"), 
        probe_isCicLoose = cms.InputTag("CicLooseMatchedSuperClusterCandsClean"), 
        probe_isCicMedium = cms.InputTag("CicMediumMatchedSuperClusterCandsClean"), 
        probe_isCicTight = cms.InputTag("CicTightMatchedSuperClusterCandsClean"), 
        probe_isCicSuperTight = cms.InputTag("CicSuperTightMatchedSuperClusterCandsClean"), 
        probe_isCicHyperTight1 = cms.InputTag("CicHyperTight1MatchedSuperClusterCandsClean"), 
        probe_isCicHyperTight2 = cms.InputTag("CicHyperTight2MatchedSuperClusterCandsClean"), 
        probe_isCicHyperTight3 = cms.InputTag("CicHyperTight3MatchedSuperClusterCandsClean"), 
        probe_isCicHyperTight4 = cms.InputTag("CicHyperTight4MatchedSuperClusterCandsClean"),        
        probe_passingHLT = cms.InputTag("TagMatchedSuperClusterCandsClean")
    ),
    probeMatches  = cms.InputTag("McMatchSC"),
    allProbes     = cms.InputTag("goodSuperClustersClean")
)
process.SuperClusterToGsfElectron.variables.probe_dRjet = cms.InputTag("superClusterDRToNearestJet")
process.SuperClusterToGsfElectron.variables.probe_nJets = cms.InputTag("JetMultiplicityInSCEvents")
process.SuperClusterToGsfElectron.variables.probe_dist = cms.InputTag("SCConvRejVars","dist")
process.SuperClusterToGsfElectron.variables.probe_dcot = cms.InputTag("SCConvRejVars","dcot")
process.SuperClusterToGsfElectron.variables.probe_convradius = cms.InputTag("SCConvRejVars","convradius")
process.SuperClusterToGsfElectron.variables.probe_passConvRej = cms.InputTag("SCConvRejVars","passConvRej")
process.SuperClusterToGsfElectron.tagVariables.dRjet = cms.InputTag("GsfDRToNearestJet")
process.SuperClusterToGsfElectron.tagVariables.nJets = cms.InputTag("JetMultiplicityInGsfEvents")
process.SuperClusterToGsfElectron.tagVariables.eidCicVeryLoose = cms.InputTag("eidVeryLoose")
process.SuperClusterToGsfElectron.tagVariables.eidCicLoose = cms.InputTag("eidLoose")
process.SuperClusterToGsfElectron.tagVariables.eidCicMedium = cms.InputTag("eidMedium")
process.SuperClusterToGsfElectron.tagVariables.eidCicTight = cms.InputTag("eidTight")
process.SuperClusterToGsfElectron.tagVariables.eidCicSuperTight = cms.InputTag("eidSuperTight")
process.SuperClusterToGsfElectron.tagVariables.eidCicHyperTight1 = cms.InputTag("eidHyperTight1")
process.SuperClusterToGsfElectron.tagVariables.eidCicHyperTight2 = cms.InputTag("eidHyperTight2")
process.SuperClusterToGsfElectron.tagVariables.eidCicHyperTight3 = cms.InputTag("eidHyperTight3")
process.SuperClusterToGsfElectron.tagVariables.eidCicHyperTight4 = cms.InputTag("eidHyperTight4")
process.SuperClusterToGsfElectron.tagVariables.eidLikelihood = cms.InputTag("eidLikelihoodExt")
process.SuperClusterToGsfElectron.tagVariables.dist = cms.InputTag("GsfConvRejVars","dist")
process.SuperClusterToGsfElectron.tagVariables.dcot = cms.InputTag("GsfConvRejVars","dcot")
process.SuperClusterToGsfElectron.tagVariables.convradius = cms.InputTag("GsfConvRejVars","convradius")
process.SuperClusterToGsfElectron.tagVariables.passConvRej = cms.InputTag("GsfConvRejVars","passConvRej")



## good photon --> gsf electron
process.PhotonToGsfElectron = process.SuperClusterToGsfElectron.clone()
process.PhotonToGsfElectron.tagProbePairs = cms.InputTag("tagPhoton")
process.PhotonToGsfElectron.flags = cms.PSet(
    probe_passingGsf = cms.InputTag("GsfMatchedPhotonCands"),
    probe_passingHLT = cms.InputTag("TagMatchedPhotonCands"),
    probe_isWP95 = cms.InputTag("WP95MatchedPhotonCands"),
    probe_isWP90 = cms.InputTag("WP90MatchedPhotonCands"),
    probe_isWP85 = cms.InputTag("WP85MatchedPhotonCands"),        
    probe_isWP80 = cms.InputTag("WP80MatchedPhotonCands"),
    probe_isWP70 = cms.InputTag("WP70MatchedPhotonCands"),
    probe_isWP60 = cms.InputTag("WP60MatchedPhotonCands"),
    probe_isCicVeryLoose = cms.InputTag("CicVeryLooseMatchedPhotonCands"), 
    probe_isCicLoose = cms.InputTag("CicLooseMatchedPhotonCands"), 
    probe_isCicMedium = cms.InputTag("CicMediumMatchedPhotonCands"), 
    probe_isCicTight = cms.InputTag("CicTightMatchedPhotonCands"), 
    probe_isCicSuperTight = cms.InputTag("CicSuperTightMatchedPhotonCands"), 
    probe_isCicHyperTight1 = cms.InputTag("CicHyperTight1MatchedPhotonCands"), 
    probe_isCicHyperTight2 = cms.InputTag("CicHyperTight2MatchedPhotonCands"), 
    probe_isCicHyperTight3 = cms.InputTag("CicHyperTight3MatchedPhotonCands"), 
    probe_isCicHyperTight4 = cms.InputTag("CicHyperTight4MatchedPhotonCands")        
    )
process.PhotonToGsfElectron.probeMatches  = cms.InputTag("McMatchPhoton")
process.PhotonToGsfElectron.allProbes     = cms.InputTag("goodPhotons")
process.PhotonToGsfElectron.variables.probe_dRjet = cms.InputTag("PhotonDRToNearestJet")
process.PhotonToGsfElectron.variables.probe_nJets = cms.InputTag("JetMultiplicityInPhotonEvents")
process.PhotonToGsfElectron.variables.probe_trackiso = cms.string("trkSumPtHollowConeDR03")
process.PhotonToGsfElectron.variables.probe_ecaliso = cms.string("ecalRecHitSumEtConeDR03")
process.PhotonToGsfElectron.variables.probe_hcaliso = cms.string("hcalTowerSumEtConeDR03")
process.PhotonToGsfElectron.variables.probe_HoverE  = cms.string("hadronicOverEm")
process.PhotonToGsfElectron.variables.probe_sigmaIetaIeta = cms.string("sigmaIetaIeta")
process.PhotonToGsfElectron.variables.probe_dist = cms.InputTag("PhotonConvRejVars","dist")
process.PhotonToGsfElectron.variables.probe_dcot = cms.InputTag("PhotonConvRejVars","dcot")
process.PhotonToGsfElectron.variables.probe_convradius = cms.InputTag("PhotonConvRejVars","convradius")
process.PhotonToGsfElectron.variables.probe_passConvRej = cms.InputTag("PhotonConvRejVars","passConvRej")
process.PhotonToGsfElectron.tagVariables.dRjet = cms.InputTag("GsfDRToNearestJet")
process.PhotonToGsfElectron.tagVariables.nJets = cms.InputTag("JetMultiplicityInGsfEvents")
process.PhotonToGsfElectron.tagVariables.eidCicVeryLoose = cms.InputTag("eidVeryLoose")
process.PhotonToGsfElectron.tagVariables.eidCicLoose = cms.InputTag("eidLoose")
process.PhotonToGsfElectron.tagVariables.eidCicMedium = cms.InputTag("eidMedium")
process.PhotonToGsfElectron.tagVariables.eidCicTight = cms.InputTag("eidTight")
process.PhotonToGsfElectron.tagVariables.eidCicSuperTight = cms.InputTag("eidSuperTight")
process.PhotonToGsfElectron.tagVariables.eidCicHyperTight1 = cms.InputTag("eidHyperTight1")
process.PhotonToGsfElectron.tagVariables.eidCicHyperTight2 = cms.InputTag("eidHyperTight2")
process.PhotonToGsfElectron.tagVariables.eidCicHyperTight3 = cms.InputTag("eidHyperTight3")
process.PhotonToGsfElectron.tagVariables.eidCicHyperTight4 = cms.InputTag("eidHyperTight4")
process.PhotonToGsfElectron.tagVariables.eidLikelihood = cms.InputTag("eidLikelihoodExt")
process.PhotonToGsfElectron.tagVariables.dist = cms.InputTag("GsfConvRejVars","dist")
process.PhotonToGsfElectron.tagVariables.dcot = cms.InputTag("GsfConvRejVars","dcot")
process.PhotonToGsfElectron.tagVariables.convradius = cms.InputTag("GsfConvRejVars","convradius")
process.PhotonToGsfElectron.tagVariables.passConvRej = cms.InputTag("GsfConvRejVars","passConvRej")

##   ____      __       __    ___                 ___    _ 
##  / ___|___ / _|      \ \  |_ _|___  ___       |_ _|__| |
## | |  _/ __| |_   _____\ \  | |/ __|/ _ \       | |/ _` |
## | |_| \__ \  _| |_____/ /  | |\__ \ (_) |  _   | | (_| |
##  \____|___/_|        /_/  |___|___/\___/  ( ) |___\__,_|
##                                           |/            
##  gsf electron --> isolation, electron id  etc.
process.GsfElectronToId = cms.EDAnalyzer("TagProbeFitTreeProducer",
    mcTruthCommonStuff, CommonStuffForGsfElectronProbe,                        
    tagProbePairs = cms.InputTag("tagGsf"),
    arbitration   = cms.string("Random2"),
    flags = cms.PSet(
        probe_isWP95 = cms.InputTag("PassingWP95"),
        probe_isWP90 = cms.InputTag("PassingWP90"),
        probe_isWP85 = cms.InputTag("PassingWP85"),        
        probe_isWP80 = cms.InputTag("PassingWP80"),
        probe_isWP70 = cms.InputTag("PassingWP70"),
        probe_isWP60 = cms.InputTag("PassingWP60"),
        probe_isCicVeryLoose = cms.InputTag("PassingCicVeryLoose"),
        probe_isCicLoose = cms.InputTag("PassingCicLoose"),
        probe_isCicMedium = cms.InputTag("PassingCicMedium"),
        probe_isCicTight = cms.InputTag("PassingCicTight"),
        probe_isCicSuperTight = cms.InputTag("PassingCicSuperTight"),          
        probe_isCicHyperTight1 = cms.InputTag("PassingCicHyperTight1"),
        probe_isCicHyperTight2 = cms.InputTag("PassingCicHyperTight2"),
        probe_isCicHyperTight3 = cms.InputTag("PassingCicHyperTight3"),
        probe_isCicHyperTight4 = cms.InputTag("PassingCicHyperTight4"),   
        probe_passingHLT = cms.InputTag("PassingHLT")        
    ),
    probeMatches  = cms.InputTag("McMatchGsf"),
    allProbes     = cms.InputTag("goodElectrons")
)
process.GsfElectronToId.variables.probe_dRjet = cms.InputTag("GsfDRToNearestJet")
process.GsfElectronToId.variables.probe_nJets = cms.InputTag("JetMultiplicityInGsfEvents")
process.GsfElectronToId.variables.probe_eidCicVeryLoose = cms.InputTag("eidVeryLoose")
process.GsfElectronToId.variables.probe_eidCicLoose = cms.InputTag("eidLoose")
process.GsfElectronToId.variables.probe_eidCicMedium = cms.InputTag("eidMedium")
process.GsfElectronToId.variables.probe_eidCicTight = cms.InputTag("eidTight")
process.GsfElectronToId.variables.probe_eidCicSuperTight = cms.InputTag("eidSuperTight")
process.GsfElectronToId.variables.probe_eidCicHyperTight1 = cms.InputTag("eidHyperTight1")
process.GsfElectronToId.variables.probe_eidCicHyperTight2 = cms.InputTag("eidHyperTight2")
process.GsfElectronToId.variables.probe_eidCicHyperTight3 = cms.InputTag("eidHyperTight3")
process.GsfElectronToId.variables.probe_eidCicHyperTight4 = cms.InputTag("eidHyperTight4")
process.GsfElectronToId.variables.probe_eidLikelihood = cms.InputTag("eidLikelihoodExt")
process.GsfElectronToId.variables.probe_dist = cms.InputTag("GsfConvRejVars","dist")
process.GsfElectronToId.variables.probe_dcot = cms.InputTag("GsfConvRejVars","dcot")
process.GsfElectronToId.variables.probe_convradius = cms.InputTag("GsfConvRejVars","convradius")
process.GsfElectronToId.variables.probe_passConvRej = cms.InputTag("GsfConvRejVars","passConvRej")
process.GsfElectronToId.tagVariables.dRjet = cms.InputTag("GsfDRToNearestJet")
process.GsfElectronToId.tagVariables.nJets = cms.InputTag("JetMultiplicityInGsfEvents")
process.GsfElectronToId.tagVariables.eidCicVeryLoose = cms.InputTag("eidVeryLoose")
process.GsfElectronToId.tagVariables.eidCicLoose = cms.InputTag("eidLoose")
process.GsfElectronToId.tagVariables.eidCicMedium = cms.InputTag("eidMedium")
process.GsfElectronToId.tagVariables.eidCicTight = cms.InputTag("eidTight")
process.GsfElectronToId.tagVariables.eidCicSuperTight = cms.InputTag("eidSuperTight")
process.GsfElectronToId.tagVariables.eidCicHyperTight1 = cms.InputTag("eidHyperTight1")
process.GsfElectronToId.tagVariables.eidCicHyperTight2 = cms.InputTag("eidHyperTight2")
process.GsfElectronToId.tagVariables.eidCicHyperTight3 = cms.InputTag("eidHyperTight3")
process.GsfElectronToId.tagVariables.eidCicHyperTight4 = cms.InputTag("eidHyperTight4")
process.GsfElectronToId.tagVariables.eidLikelihood = cms.InputTag("eidLikelihoodExt")
process.GsfElectronToId.tagVariables.dist = cms.InputTag("GsfConvRejVars","dist")
process.GsfElectronToId.tagVariables.dcot = cms.InputTag("GsfConvRejVars","dcot")
process.GsfElectronToId.tagVariables.convradius = cms.InputTag("GsfConvRejVars","convradius")
process.GsfElectronToId.tagVariables.passConvRej = cms.InputTag("GsfConvRejVars","passConvRej")
process.GsfElectronToId.pairVariables.costheta = cms.InputTag("CSVarsTagGsf","costheta")
process.GsfElectronToId.pairVariables.sin2theta = cms.InputTag("CSVarsTagGsf","sin2theta")
process.GsfElectronToId.pairVariables.tanphi = cms.InputTag("CSVarsTagGsf","tanphi")


process.GsfElectronPlusGsfElectron = process.GsfElectronToId.clone()
process.GsfElectronPlusGsfElectron.tagProbePairs = cms.InputTag("GsfGsf")
process.GsfElectronPlusGsfElectron.tagMatches = cms.InputTag("McMatchGsf")
process.GsfElectronPlusGsfElectron.pairVariables.costheta = cms.InputTag("CSVarsGsfGsf","costheta")
process.GsfElectronPlusGsfElectron.pairVariables.sin2theta = cms.InputTag("CSVarsGsfGsf","sin2theta")
process.GsfElectronPlusGsfElectron.pairVariables.tanphi = cms.InputTag("CSVarsGsfGsf","tanphi")
if MC_flag:
    process.GsfElectronPlusGsfElectron.PUWeightSrc = cms.InputTag("pileupReweightingProducer","pileupWeights")


process.GsfElectronPlusMet = process.GsfElectronToId.clone()
process.GsfElectronPlusMet.tagProbePairs = cms.InputTag("elecMet")
process.GsfElectronPlusMet.tagVariables = cms.PSet()
process.GsfElectronPlusMet.pairVariables =  cms.PSet(ZVariablesToStore)
process.GsfElectronPlusMet.pairFlags =  cms.PSet( isMTabove40 = cms.string("mt > 40") )
process.GsfElectronPlusMet.isMC = cms.bool(False)


##    ___    _       __    _   _ _   _____ 
##   |_ _|__| |      \ \  | | | | | |_   _|
##    | |/ _` |  _____\ \ | |_| | |   | |  
##    | | (_| | |_____/ / |  _  | |___| |  
##   |___\__,_|      /_/  |_| |_|_____|_|
##
##  offline selection --> HLT. First specify which quantities to store in the TP tree. 
if MC_flag:
    HLTmcTruthCommonStuff = cms.PSet(
        isMC = cms.bool(MC_flag),
        tagMatches = cms.InputTag("McMatchTag"),
        motherPdgId = cms.vint32(22,23),
        makeMCUnbiasTree = cms.bool(MC_flag),
        checkMotherInUnbiasEff = cms.bool(MC_flag),
        mcVariables = cms.PSet(
          probe_eta = cms.string("eta"),
          probe_phi  = cms.string("phi"),
          probe_et  = cms.string("et"),
          probe_charge = cms.string("charge"),
        ),
        mcFlags     =  cms.PSet(
          probe_flag = cms.string("pt>0")
        ),      
        )
else:
     HLTmcTruthCommonStuff = cms.PSet(
         isMC = cms.bool(False)
         )

##  WP95 --> HLT
process.WP95ToHLT = cms.EDAnalyzer("TagProbeFitTreeProducer",
    HLTmcTruthCommonStuff,                                
    variables = cms.PSet(
      probe_gsfEle_eta = cms.string("eta"),
      probe_gsfEle_phi  = cms.string("phi"),
      probe_gsfEle_et  = cms.string("et"),
      probe_gsfEle_charge = cms.string("charge"),
      probe_sc_et    = cms.string("superCluster.energy*sin(superClusterPosition.theta)"),    
      probe_sc_eta    = cms.string("superCluster.eta"), 
      probe_sc_phi    = cms.string("superCluster.phi"),
      probe_gsfEle_isEB           = cms.string("isEB"),
      probe_gsfEle_isEE           = cms.string("isEE"),
      probe_gsfEle_isGap          = cms.string("isGap"),
    ),
    addRunLumiInfo   =  cms.bool (False),
    addEventVariablesInfo   =  cms.bool (False),                                                        
    tagProbePairs = cms.InputTag("tagWP95"),
    arbitration   = cms.string("Random2"),
    flags = cms.PSet( 
        probe_passingHLT = cms.InputTag("PassingHLT")        
    ),
    probeMatches  = cms.InputTag("McMatchWP95"),
    allProbes     = cms.InputTag("PassingWP95")
)
if MC_flag:
    process.WP95ToHLT.PUWeightSrc = cms.InputTag("pileupReweightingProducer","pileupWeights")


##  WP90 --> HLT
process.WP90ToHLT = process.WP95ToHLT.clone()
process.WP90ToHLT.tagProbePairs = cms.InputTag("tagWP90")
process.WP90ToHLT.probeMatches  = cms.InputTag("McMatchWP90")
process.WP90ToHLT.allProbes     = cms.InputTag("PassingWP90")

##  WP85 --> HLT
process.WP85ToHLT = process.WP95ToHLT.clone()
process.WP85ToHLT.tagProbePairs = cms.InputTag("tagWP85")
process.WP85ToHLT.probeMatches  = cms.InputTag("McMatchWP85")
process.WP85ToHLT.allProbes     = cms.InputTag("PassingWP85")

##  WP80 --> HLT
process.WP80ToHLT = process.WP95ToHLT.clone()
process.WP80ToHLT.tagProbePairs = cms.InputTag("tagWP80")
process.WP80ToHLT.probeMatches  = cms.InputTag("McMatchWP80")
process.WP80ToHLT.allProbes     = cms.InputTag("PassingWP80")

##  WP70 --> HLT
process.WP70ToHLT = process.WP95ToHLT.clone()
process.WP70ToHLT.tagProbePairs = cms.InputTag("tagWP70")
process.WP70ToHLT.probeMatches  = cms.InputTag("McMatchWP70")
process.WP70ToHLT.allProbes     = cms.InputTag("PassingWP70")

##  WP60 --> HLT
process.WP60ToHLT = process.WP95ToHLT.clone()
process.WP60ToHLT.tagProbePairs = cms.InputTag("tagWP60")
process.WP60ToHLT.probeMatches  = cms.InputTag("McMatchWP60")
process.WP60ToHLT.allProbes     = cms.InputTag("PassingWP60")

##  CicVeryLoose --> HLT
process.CicVeryLooseToHLT = process.WP95ToHLT.clone()
process.CicVeryLooseToHLT.tagProbePairs = cms.InputTag("tagCicVeryLoose")
process.CicVeryLooseToHLT.probeMatches  = cms.InputTag("McMatchCicVeryLoose")
process.CicVeryLooseToHLT.allProbes     = cms.InputTag("PassingCicVeryLoose")

##  CicLoose --> HLT
process.CicLooseToHLT = process.WP95ToHLT.clone()
process.CicLooseToHLT.tagProbePairs = cms.InputTag("tagCicLoose")
process.CicLooseToHLT.probeMatches  = cms.InputTag("McMatchCicLoose")
process.CicLooseToHLT.allProbes     = cms.InputTag("PassingCicLoose")

##  CicMedium --> HLT
process.CicMediumToHLT = process.WP95ToHLT.clone()
process.CicMediumToHLT.tagProbePairs = cms.InputTag("tagCicMedium")
process.CicMediumToHLT.probeMatches  = cms.InputTag("McMatchCicMedium")
process.CicMediumToHLT.allProbes     = cms.InputTag("PassingCicMedium")

##  CicTight --> HLT
process.CicTightToHLT = process.WP95ToHLT.clone()
process.CicTightToHLT.tagProbePairs = cms.InputTag("tagCicTight")
process.CicTightToHLT.probeMatches  = cms.InputTag("McMatchCicTight")
process.CicTightToHLT.allProbes     = cms.InputTag("PassingCicTight")

##  CicSuperTight --> HLT
process.CicSuperTightToHLT = process.WP95ToHLT.clone()
process.CicSuperTightToHLT.tagProbePairs = cms.InputTag("tagCicSuperTight")
process.CicSuperTightToHLT.probeMatches  = cms.InputTag("McMatchCicSuperTight")
process.CicSuperTightToHLT.allProbes     = cms.InputTag("PassingCicSuperTight")

##  CicHyperTight1 --> HLT
process.CicHyperTight1ToHLT = process.WP95ToHLT.clone()
process.CicHyperTight1ToHLT.tagProbePairs = cms.InputTag("tagCicHyperTight1")
process.CicHyperTight1ToHLT.probeMatches  = cms.InputTag("McMatchCicHyperTight1")
process.CicHyperTight1ToHLT.allProbes     = cms.InputTag("PassingCicHyperTight1")

##  CicHyperTight2 --> HLT
process.CicHyperTight2ToHLT = process.WP95ToHLT.clone()
process.CicHyperTight2ToHLT.tagProbePairs = cms.InputTag("tagCicHyperTight2")
process.CicHyperTight2ToHLT.probeMatches  = cms.InputTag("McMatchCicHyperTight2")
process.CicHyperTight2ToHLT.allProbes     = cms.InputTag("PassingCicHyperTight2")

##  CicHyperTight3 --> HLT
process.CicHyperTight3ToHLT = process.WP95ToHLT.clone()
process.CicHyperTight3ToHLT.tagProbePairs = cms.InputTag("tagCicHyperTight3")
process.CicHyperTight3ToHLT.probeMatches  = cms.InputTag("McMatchCicHyperTight3")
process.CicHyperTight3ToHLT.allProbes     = cms.InputTag("PassingCicHyperTight3")

##  CicHyperTight4 --> HLT
process.CicHyperTight4ToHLT = process.WP95ToHLT.clone()
process.CicHyperTight4ToHLT.tagProbePairs = cms.InputTag("tagCicHyperTight4")
process.CicHyperTight4ToHLT.probeMatches  = cms.InputTag("McMatchCicHyperTight4")
process.CicHyperTight4ToHLT.allProbes     = cms.InputTag("PassingCicHyperTight4")


process.tree_sequence = cms.Sequence(
    process.SuperClusterToGsfElectron +
    process.PhotonToGsfElectron +
    process.GsfElectronToId +
    process.GsfElectronPlusGsfElectron +
    process.GsfElectronPlusMet + 
    process.WP95ToHLT +
    process.WP90ToHLT +
    process.WP85ToHLT + 
    process.WP80ToHLT +
    process.WP70ToHLT + 
    process.WP60ToHLT +
    process.CicVeryLooseToHLT +
    process.CicLooseToHLT +
    process.CicMediumToHLT +
    process.CicTightToHLT +
    process.CicSuperTightToHLT +
    process.CicHyperTight1ToHLT +
    process.CicHyperTight2ToHLT +
    process.CicHyperTight3ToHLT +
    process.CicHyperTight4ToHLT        
)    

##    ____       _   _     
##   |  _ \ __ _| |_| |__  
##   | |_) / _` | __| '_ \ 
##   |  __/ (_| | |_| | | |
##   |_|   \__,_|\__|_| |_|
##

if MC_flag:
    process.tagAndProbe = cms.Path(
        process.sc_sequence + process.eIDSequence + process.ele_sequence + 
        process.ext_ToNearestJet_sequence + 
        process.allTagsAndProbes + process.pileupReweightingProducer + 
        process.mc_sequence + 
        process.tree_sequence
        )
else:
    process.tagAndProbe = cms.Path(
        process.sc_sequence + process.eIDSequence + process.ele_sequence + 
        process.ext_ToNearestJet_sequence + 
        process.allTagsAndProbes +
        process.tree_sequence
        )
    
process.TFileService = cms.Service(
    "TFileService", fileName = cms.string( OUTPUT_FILE_NAME )
    )
