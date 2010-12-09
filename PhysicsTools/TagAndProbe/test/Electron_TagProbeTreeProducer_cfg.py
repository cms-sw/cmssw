import FWCore.ParameterSet.Config as cms

##                      _              _       
##   ___ ___  _ __  ___| |_ __ _ _ __ | |_ ___ 
##  / __/ _ \| '_ \/ __| __/ _` | '_ \| __/ __|
## | (_| (_) | | | \__ \ || (_| | | | | |_\__ \
##  \___\___/|_| |_|___/\__\__,_|_| |_|\__|___/
##                                              
########################
MC_flag = False
GLOBAL_TAG = 'GR_R_39X_V4::All'
HLTPath = "HLT_Ele17_SW_TightEleId_L1R"
OUTPUT_FILE_NAME = "testNewWrite.root" 
#HLTPath = "HLT_Ele15_SW_CaloEleId_L1R"
#HLTPath = "HLT_Ele15_SW_L1R"
#HLTPath = "HLT_Ele15_LW_L1R"
#HLTPath = "HLT_Photon15_Cleaned_L1R"

ELECTRON_ET_CUT_MIN = 10.0
ELECTRON_COLL = "gsfElectrons"
ELECTRON_CUTS = "ecalDrivenSeed==1 && (abs(superCluster.eta)<2.5) && !(1.4442<abs(superCluster.eta)<1.566) && (ecalEnergy*sin(superClusterPosition.theta)>" + str(ELECTRON_ET_CUT_MIN) + ")"
####

PHOTON_COLL = "photons"
PHOTON_CUTS = "hadronicOverEm<0.15 && (abs(superCluster.eta)<2.5) && !(1.4442<abs(superCluster.eta)<1.566) && ((isEB && sigmaIetaIeta<0.01) || (isEE && sigmaIetaIeta<0.03)) && (superCluster.energy*sin(superCluster.position.theta)>" + str(ELECTRON_ET_CUT_MIN) + ")"
####

SUPERCLUSTER_COLL_EB = "hybridSuperClusters"
SUPERCLUSTER_COLL_EE = "multi5x5SuperClustersWithPreshower"
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
process.GlobalTag.globaltag = GLOBAL_TAG
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
#process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

##   ____             _ ____                           
##  |  _ \ ___   ___ | / ___|  ___  _   _ _ __ ___ ___ 
##  | |_) / _ \ / _ \| \___ \ / _ \| | | | '__/ __/ _ \
##  |  __/ (_) | (_) | |___) | (_) | |_| | | | (_|  __/
##  |_|   \___/ \___/|_|____/ \___/ \__,_|_|  \___\___|
##  
process.source = cms.Source("PoolSource", 
    fileNames = cms.untracked.vstring(
       '/store/data/Run2010B/Electron/RECO/PromptReco-v2/000/147/454/ACDEDA3C-B7D3-DF11-A7A1-0030487C6A66.root',
       '/store/data/Run2010B/Electron/RECO/PromptReco-v2/000/147/454/223CD93D-B7D3-DF11-885E-0030487CD7B4.root',
       '/store/data/Run2010B/Electron/RECO/PromptReco-v2/000/147/453/EAB3E588-B6D3-DF11-8BDC-0030487A3232.root',
       '/store/data/Run2010B/Electron/RECO/PromptReco-v2/000/147/453/AA0C5537-B7D3-DF11-9194-0030487CD7C6.root',
       '/store/data/Run2010B/Electron/RECO/PromptReco-v2/000/147/453/A28CBA36-B7D3-DF11-9F37-00304879BAB2.root',
       '/store/data/Run2010B/Electron/RECO/PromptReco-v2/000/147/453/8C67199B-B1D3-DF11-AAC4-0030487CD7CA.root',
       '/store/data/Run2010B/Electron/RECO/PromptReco-v2/000/147/453/823B32EE-B7D3-DF11-B2CB-0030487CAF0E.root',
       '/store/data/Run2010B/Electron/RECO/PromptReco-v2/000/147/453/3C66014F-B2D3-DF11-9E18-0030487CD6DA.root',
       '/store/data/Run2010B/Electron/RECO/PromptReco-v2/000/147/453/3AA99C36-B7D3-DF11-BB90-0030487CAEAC.root',
       '/store/data/Run2010B/Electron/RECO/PromptReco-v2/000/147/453/26B82C89-B6D3-DF11-9584-0030487CD6B4.root',
       '/store/data/Run2010B/Electron/RECO/PromptReco-v2/000/147/453/0AA663A1-B8D3-DF11-ADD8-0030487CD6B4.root',
       '/store/data/Run2010B/Electron/RECO/PromptReco-v2/000/147/452/F08E2485-95D3-DF11-842A-0030486780B8.root',
       '/store/data/Run2010B/Electron/RECO/PromptReco-v2/000/147/452/DCF42686-95D3-DF11-8DF0-0030487CD76A.root',
       '/store/data/Run2010B/Electron/RECO/PromptReco-v2/000/147/452/5E17B94C-9DD3-DF11-A952-001617E30F58.root',
       '/store/data/Run2010B/Electron/RECO/PromptReco-v2/000/147/452/54CDACD8-94D3-DF11-B6A6-001617E30D12.root',
       '/store/data/Run2010B/Electron/RECO/PromptReco-v2/000/147/452/346D9037-96D3-DF11-88DF-001617C3B710.root',
       '/store/data/Run2010B/Electron/RECO/PromptReco-v2/000/147/452/029B8885-95D3-DF11-B1FE-001617E30D4A.root',    
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
    " && (gsfTrack.trackerExpectedHitsInner.numberOfHits <= 1)"
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
    " && (gsfTrack.trackerExpectedHitsInner.numberOfHits==0 && !(-0.02<convDist<0.02 && -0.02<convDcot<0.02))"
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
    " && (gsfTrack.trackerExpectedHitsInner.numberOfHits==0 && !(-0.02<convDist<0.02 && -0.02<convDcot<0.02))"
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
    " && (gsfTrack.trackerExpectedHitsInner.numberOfHits==0 && !(-0.02<convDist<0.02 && -0.02<convDcot<0.02))"
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
    " && (gsfTrack.trackerExpectedHitsInner.numberOfHits==0 && !(-0.02<convDist<0.02 && -0.02<convDcot<0.02))"
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
    " && (gsfTrack.trackerExpectedHitsInner.numberOfHits==0 && !(-0.02<convDist<0.02 && -0.02<convDcot<0.02))"
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
process.load("RecoEgamma.ElectronIdentification.cutsInCategoriesElectronIdentificationV06_cfi")
process.eIDSequence = cms.Sequence(
    process.eidVeryLoose+
    process.eidLoose+                
    process.eidMedium+ process.eidTight+
    process.eidSuperTight+ process.eidHyperTight1
    )
                         
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
    hltTags = cms.VInputTag(cms.InputTag(HLTPath,"","HLT")),
    triggerEventTag = cms.untracked.InputTag("hltTriggerSummaryAOD","","HLT"),
    triggerResultsTag = cms.untracked.InputTag("TriggerResults","","HLT")                               
)


##    _____      _                        _  __     __             
##   | ____|_  _| |_ ___ _ __ _ __   __ _| | \ \   / /_ _ _ __ ___ 
##   |  _| \ \/ / __/ _ \ '__| '_ \ / _` | |  \ \ / / _` | '__/ __|
##   | |___ >  <| ||  __/ |  | | | | (_| | |   \ V / (_| | |  \__ \
##   |_____/_/\_\\__\___|_|  |_| |_|\__,_|_|    \_/ \__,_|_|  |___/
##   
## Here we show how to use a module to compute an external variable
## process.load("JetMETCorrections.Configuration.DefaultJEC_cff")
## ak5PFResidual.useCondDB = False

process.superClusterDRToNearestJet = cms.EDProducer("DeltaRNearestJetComputer",
    probes = cms.InputTag("goodSuperClusters"),
       # ^^--- NOTA BENE: if probes are defined by ref, as in this case, 
       #       this must be the full collection, not the subset by refs.
    objects = cms.InputTag(JET_COLL),
    objectSelection = cms.string(JET_CUTS + " && pt > 15.0"),
)
process.JetMultiplicityInSCEvents = cms.EDProducer("CandMultiplicityCounter",
    probes = cms.InputTag("goodSuperClusters"),
    objects = cms.InputTag(JET_COLL),
    objectSelection = cms.string(JET_CUTS + " && pt > 15.0"),
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
    process.PassingHLT + process.Tag +
    process.TagMatchedSuperClusterCandsClean +
    process.TagMatchedPhotonCands +
    process.WP95MatchedSuperClusterCandsClean +
    process.WP90MatchedSuperClusterCandsClean +
    process.WP85MatchedSuperClusterCandsClean +
    process.WP80MatchedSuperClusterCandsClean +
    process.WP70MatchedSuperClusterCandsClean +
    process.WP60MatchedSuperClusterCandsClean +    
    process.WP95MatchedPhotonCands +
    process.WP90MatchedPhotonCands +
    process.WP85MatchedPhotonCands +
    process.WP80MatchedPhotonCands +
    process.WP70MatchedPhotonCands +
    process.WP60MatchedPhotonCands 
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
process.tagWP80 = process.tagSC.clone()
process.tagWP80.decay = cms.string("Tag PassingWP80")

process.CSVarsTagGsf = cms.EDProducer("ColinsSoperVariablesComputer",
    parentBoson = cms.InputTag("tagGsf")
)
process.CSVarsGsfGsf = process.CSVarsTagGsf.clone()
process.CSVarsGsfGsf.parentBoson = cms.InputTag("GsfGsf")
process.CSVarsTagWP95 = process.CSVarsTagGsf.clone()
process.CSVarsTagWP95.parentBoson = cms.InputTag("tagWP95")
process.CSVarsTagWP80 = process.CSVarsTagGsf.clone()
process.CSVarsTagWP80.parentBoson = cms.InputTag("tagWP80")

process.allTagsAndProbes = cms.Sequence(
    process.tagSC + process.tagPhoton + process.tagGsf +
    process.GsfGsf + process.tagWP95 + process.tagWP80 +
    process.CSVarsTagGsf + process.CSVarsTagWP95 +
    process.CSVarsTagWP80 + process.CSVarsGsfGsf
)

##    __  __  ____   __  __       _       _               
##   |  \/  |/ ___| |  \/  | __ _| |_ ___| |__   ___  ___ 
##   | |\/| | |     | |\/| |/ _` | __/ __| '_ \ / _ \/ __|
##   | |  | | |___  | |  | | (_| | || (__| | | |  __/\__ \
##   |_|  |_|\____| |_|  |_|\__,_|\__\___|_| |_|\___||___/
##                                                        
process.McMatchTag = cms.EDFilter("MCTruthDeltaRMatcherNew",
    matchPDGId = cms.vint32(11),
    src = cms.InputTag("Tag"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("genParticles"),
    checkCharge = cms.bool(True)
)
process.McMatchSC = cms.EDFilter("MCTruthDeltaRMatcherNew",
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
process.McMatchWP80 = process.McMatchTag.clone()
process.McMatchWP80.src = cms.InputTag("PassingWP80")


process.mc_sequence = cms.Sequence(
   process.McMatchTag + process.McMatchSC +
   process.McMatchPhoton + process.McMatchGsf + 
   process.McMatchWP95  + process.McMatchWP80
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
    probe_gsfEle_missingHits = cms.string("gsfTrack.trackerExpectedHitsInner.numberOfHits"),
    probe_gsfEle_convDist = cms.string("convDist"),
    probe_gsfEle_convDcot = cms.string("convDcot"),
    probe_gsfEle_convRadius = cms.string("convRadius"),        
    probe_gsfEle_hasValidHitInFirstPixelBarrel = cms.string("gsfTrack.hitPattern.hasValidHitInFirstPixelBarrel"),
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
    gsfEle_missingHits = cms.string("gsfTrack.trackerExpectedHitsInner.numberOfHits"),
    gsfEle_convDist = cms.string("convDist"),
    gsfEle_convDcot = cms.string("convDcot"),
    gsfEle_convRadius = cms.string("convRadius"),     
    gsfEle_hasValidHitInFirstPixelBarrel = cms.string("gsfTrack.hitPattern.hasValidHitInFirstPixelBarrel"),
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
    ignoreExceptions =  cms.bool (False),
    addRunLumiInfo   =  cms.bool (True),
    addEventVariablesInfo   =  cms.bool (True),
    pairVariables =  cms.PSet(ZVariablesToStore),
    pairFlags     =  cms.PSet(
          mass60to120 = cms.string("40 < mass < 1000")
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
    probe_isWP60 = cms.InputTag("WP60MatchedPhotonCands")        
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


##   ____      __       __    ___                 ___    _ 
##  / ___|___ / _|      \ \  |_ _|___  ___       |_ _|__| |
## | |  _/ __| |_   _____\ \  | |/ __|/ _ \       | |/ _` |
## | |_| \__ \  _| |_____/ /  | |\__ \ (_) |  _   | | (_| |
##  \____|___/_|        /_/  |___|___/\___/  ( ) |___\__,_|
##                                           |/            
##  gsf electron --> isolation, electron id  etc.
process.GsfElectronToIdToHLT = cms.EDAnalyzer("TagProbeFitTreeProducer",
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
        probe_passingHLT = cms.InputTag("PassingHLT")        
    ),
    probeMatches  = cms.InputTag("McMatchGsf"),
    allProbes     = cms.InputTag("goodElectrons")
)
process.GsfElectronToIdToHLT.variables.probe_dRjet = cms.InputTag("GsfDRToNearestJet")
process.GsfElectronToIdToHLT.variables.probe_nJets = cms.InputTag("JetMultiplicityInGsfEvents")
process.GsfElectronToIdToHLT.variables.probe_eidCicVeryLoose = cms.InputTag("eidVeryLoose")
process.GsfElectronToIdToHLT.variables.probe_eidCicLoose = cms.InputTag("eidLoose")
process.GsfElectronToIdToHLT.variables.probe_eidCicMedium = cms.InputTag("eidMedium")
process.GsfElectronToIdToHLT.variables.probe_eidCicTight = cms.InputTag("eidTight")
process.GsfElectronToIdToHLT.variables.probe_eidCicSuperTight = cms.InputTag("eidSuperTight")
process.GsfElectronToIdToHLT.variables.probe_eidCicHyperTight = cms.InputTag("eidHyperTight1")
process.GsfElectronToIdToHLT.variables.probe_dist = cms.InputTag("GsfConvRejVars","dist")
process.GsfElectronToIdToHLT.variables.probe_dcot = cms.InputTag("GsfConvRejVars","dcot")
process.GsfElectronToIdToHLT.variables.probe_convradius = cms.InputTag("GsfConvRejVars","convradius")
process.GsfElectronToIdToHLT.variables.probe_passConvRej = cms.InputTag("GsfConvRejVars","passConvRej")
process.GsfElectronToIdToHLT.tagVariables.dRjet = cms.InputTag("GsfDRToNearestJet")
process.GsfElectronToIdToHLT.tagVariables.nJets = cms.InputTag("JetMultiplicityInGsfEvents")
process.GsfElectronToIdToHLT.tagVariables.eidCicVeryLoose = cms.InputTag("eidVeryLoose")
process.GsfElectronToIdToHLT.tagVariables.eidCicLoose = cms.InputTag("eidLoose")
process.GsfElectronToIdToHLT.tagVariables.eidCicMedium = cms.InputTag("eidMedium")
process.GsfElectronToIdToHLT.tagVariables.eidCicTight = cms.InputTag("eidTight")
process.GsfElectronToIdToHLT.tagVariables.eidCicSuperTight = cms.InputTag("eidSuperTight")
process.GsfElectronToIdToHLT.tagVariables.eidCicHyperTight = cms.InputTag("eidHyperTight1")
process.GsfElectronToIdToHLT.tagVariables.dist = cms.InputTag("GsfConvRejVars","dist")
process.GsfElectronToIdToHLT.tagVariables.dcot = cms.InputTag("GsfConvRejVars","dcot")
process.GsfElectronToIdToHLT.tagVariables.convradius = cms.InputTag("GsfConvRejVars","convradius")
process.GsfElectronToIdToHLT.tagVariables.passConvRej = cms.InputTag("GsfConvRejVars","passConvRej")
process.GsfElectronToIdToHLT.pairVariables.costheta = cms.InputTag("CSVarsTagGsf","costheta")
process.GsfElectronToIdToHLT.pairVariables.sin2theta = cms.InputTag("CSVarsTagGsf","sin2theta")
process.GsfElectronToIdToHLT.pairVariables.tanphi = cms.InputTag("CSVarsTagGsf","tanphi")


process.GsfElectronPlusGsfElectron = process.GsfElectronToIdToHLT.clone()
process.GsfElectronPlusGsfElectron.tagProbePairs = cms.InputTag("GsfGsf")
process.GsfElectronPlusGsfElectron.pairVariables.costheta = cms.InputTag("CSVarsGsfGsf","costheta")
process.GsfElectronPlusGsfElectron.pairVariables.sin2theta = cms.InputTag("CSVarsGsfGsf","sin2theta")
process.GsfElectronPlusGsfElectron.pairVariables.tanphi = cms.InputTag("CSVarsGsfGsf","tanphi")

##    ___    _       __    _   _ _   _____ 
##   |_ _|__| |      \ \  | | | | | |_   _|
##    | |/ _` |  _____\ \ | |_| | |   | |  
##    | | (_| | |_____/ / |  _  | |___| |  
##   |___\__,_|      /_/  |_| |_|_____|_|
##
##  WP95 --> HLT
process.WP95ToHLT = process.GsfElectronToIdToHLT.clone()
process.WP95ToHLT.tagProbePairs = cms.InputTag("tagWP95")
process.WP95ToHLT.probeMatches  = cms.InputTag("McMatchWP95")
process.WP95ToHLT.allProbes     = cms.InputTag("PassingWP95")
process.WP95ToHLT.pairVariables.costheta = cms.InputTag("CSVarsTagWP95","costheta")
process.WP95ToHLT.pairVariables.sin2theta = cms.InputTag("CSVarsTagWP95","sin2theta")
process.WP95ToHLT.pairVariables.tanphi = cms.InputTag("CSVarsTagWP95","tanphi")



##  WP80 --> HLT
process.WP80ToHLT = process.GsfElectronToIdToHLT.clone()
process.WP80ToHLT.tagProbePairs = cms.InputTag("tagWP80")
process.WP80ToHLT.probeMatches  = cms.InputTag("McMatchWP80")
process.WP80ToHLT.allProbes     = cms.InputTag("PassingWP80")
process.WP80ToHLT.pairVariables.costheta = cms.InputTag("CSVarsTagWP80","costheta")
process.WP80ToHLT.pairVariables.sin2theta = cms.InputTag("CSVarsTagWP80","sin2theta")
process.WP80ToHLT.pairVariables.tanphi = cms.InputTag("CSVarsTagWP80","tanphi")


process.tree_sequence = cms.Sequence(
    process.SuperClusterToGsfElectron +
    process.PhotonToGsfElectron +
    process.GsfElectronToIdToHLT +
    process.GsfElectronPlusGsfElectron +
    process.WP95ToHLT +
    process.WP80ToHLT
)    

##    ____       _   _     
##   |  _ \ __ _| |_| |__  
##   | |_) / _` | __| '_ \ 
##   |  __/ (_| | |_| | | |
##   |_|   \__,_|\__|_| |_|
##
process.tagAndProbe = cms.Path(
    process.sc_sequence + process.ele_sequence + process.eIDSequence + 
    process.ext_ToNearestJet_sequence + 
    process.allTagsAndProbes +
    #process.mc_sequence + 
    process.tree_sequence
)
process.TFileService = cms.Service(
    "TFileService", fileName = cms.string( OUTPUT_FILE_NAME )
    )
