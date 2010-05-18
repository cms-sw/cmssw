import FWCore.ParameterSet.Config as cms

process = cms.Process("TagProbe")

##    ___            _           _      
##   |_ _|_ __   ___| |_   _  __| | ___ 
##    | || '_ \ / __| | | | |/ _` |/ _ \
##    | || | | | (__| | |_| | (_| |  __/
##   |___|_| |_|\___|_|\__,_|\__,_|\___|

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
#process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.MessageLogger.cerr.FwkReport.reportEvery = 100


##   ____             _ ____                           
##  |  _ \ ___   ___ | / ___|  ___  _   _ _ __ ___ ___ 
##  | |_) / _ \ / _ \| \___ \ / _ \| | | | '__/ __/ _ \
##  |  __/ (_) | (_) | |___) | (_) | |_| | | | (_|  __/
##  |_|   \___/ \___/|_|____/ \___/ \__,_|_|  \___\___|
##  

process.source = cms.Source("PoolSource", 
    fileNames = cms.untracked.vstring(
'/store/mc/Spring10/ZeeJet_Pt80to120/GEN-SIM-RECO/START3X_V26_S09-v1/0013/F44C88D8-3A47-DF11-AA60-0030487F16BF.root',
'/store/mc/Spring10/ZeeJet_Pt80to120/GEN-SIM-RECO/START3X_V26_S09-v1/0013/F07E1F27-3B47-DF11-9D50-0030487F9351.root',
'/store/mc/Spring10/ZeeJet_Pt80to120/GEN-SIM-RECO/START3X_V26_S09-v1/0013/EE363770-4247-DF11-AD74-0030487F1A4F.root',
'/store/mc/Spring10/ZeeJet_Pt80to120/GEN-SIM-RECO/START3X_V26_S09-v1/0013/E6335250-3947-DF11-A6B8-003048D3CD92.root',
'/store/mc/Spring10/ZeeJet_Pt80to120/GEN-SIM-RECO/START3X_V26_S09-v1/0013/E2758F8E-3947-DF11-A621-0030487E52A3.root',
    )
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(4000) )    


##   ____                         ____ _           _            
##  / ___| _   _ _ __   ___ _ __ / ___| |_   _ ___| |_ ___ _ __ 
##  \___ \| | | | '_ \ / _ \ '__| |   | | | | / __| __/ _ \ '__|
##   ___) | |_| | |_) |  __/ |  | |___| | |_| \__ \ ||  __/ |   
##  |____/ \__,_| .__/ \___|_|   \____|_|\__,_|___/\__\___|_|   
##  

#  SuperClusters  ################
process.superClusters = cms.EDFilter("SuperClusterMerger",
   src = cms.VInputTag(cms.InputTag("hybridSuperClusters","", "RECO"),
                       cms.InputTag("multi5x5SuperClustersWithPreshower","", "RECO"))  
)

process.superClusterCands = cms.EDProducer("ConcreteEcalCandidateProducer",
   src = cms.InputTag("superClusters"),
   particleType = cms.int32(11),
)

#   Get the above SC's Candidates and place a cut on their Et and eta
process.goodSuperClusters = cms.EDFilter("CandViewSelector",
      src = cms.InputTag("superClusterCands"),
      cut = cms.string("et>20.0 && abs(eta)<2.5 && !(1.4442< abs(eta) <1.560)"),
      filter = cms.bool(True)
)                                         
                                         

#### remove real jets (with high hadronic energy fraction) from SC collection
##### this improves the purity of the probe sample without affecting efficiency

process.myJets = cms.EDFilter("CaloJetSelector",   
    src = cms.InputTag("ak5CaloJets"),
    cut = cms.string('pt>5 && energyFractionHadronic > 0.15')
)


process.goodSuperClustersClean = cms.EDFilter("CandViewCleaner",
    srcCands = cms.InputTag("goodSuperClusters"),
    module_label = cms.string(''),
    srcObjects = cms.VInputTag(cms.InputTag("myJets")),
    deltaRMin = cms.double(0.1)
)



## process.superClusters = cms.EDFilter("EgammaHLTRecoEcalCandidateProducers",
##    scHybridBarrelProducer =  cms.InputTag("hybridSuperClusters","", "RECO"),
##    scIslandEndcapProducer =  cms.InputTag("multi5x5SuperClustersWithPreshower","", "RECO"),    
##    recoEcalCandidateCollection = cms.string("")
## )


process.sc_sequence = cms.Sequence( process.superClusters *
                                    process.superClusterCands *
                                    process.goodSuperClusters *
                                    process.myJets * process.goodSuperClustersClean)


##    ____      __ _____ _           _                   
##   / ___|___ / _| ____| | ___  ___| |_ _ __ ___  _ __  
##  | |  _/ __| |_|  _| | |/ _ \/ __| __| '__/ _ \| '_ \ 
##  | |_| \__ \  _| |___| |  __/ (__| |_| | | (_) | | | |
##   \____|___/_| |_____|_|\___|\___|\__|_|  \___/|_| |_|
##  

#  GsfElectron ################ 
process.PassingGsf = cms.EDFilter("GsfElectronRefSelector",
    src = cms.InputTag("gsfElectrons"),
    cut = cms.string("(abs(superCluster.eta)<2.5) && !(1.4442<abs(superCluster.eta)<1.560)"
                     " && (ecalEnergy*sin(superClusterPosition.theta)>20.0)")    
)


process.GsfMatchedSuperClusterCands = cms.EDProducer("ElectronMatchedCandidateProducer",
   src     = cms.InputTag("goodSuperClusters"),
   ReferenceElectronCollection = cms.untracked.InputTag("PassingGsf"),
   deltaR =  cms.untracked.double(0.3)
)
     

##     ___           _       _   _             
##    |_ _|___  ___ | | __ _| |_(_) ___  _ __  
##     | |/ __|/ _ \| |/ _` | __| |/ _ \| '_ \ 
##     | |\__ \ (_) | | (_| | |_| | (_) | | | |
##    |___|___/\___/|_|\__,_|\__|_|\___/|_| |_|

                                         
#  Isolation ################ 
process.PassingIsolation = cms.EDFilter("GsfElectronRefSelector",
    src = cms.InputTag("gsfElectrons"),
    cut = cms.string(process.PassingGsf.cut.value() +
         " && ( isEB && (dr03TkSumPt<7.2) && (dr04EcalRecHitSumEt<5.7) && (dr04HcalTowerSumEt<8.1))"
         " || (isEE && (dr03TkSumPt<5.1) && (dr04EcalRecHitSumEt<5.0) && (dr04HcalTowerSumEt<3.4))")
)

##    _____ _           _                     ___    _ 
##   | ____| | ___  ___| |_ _ __ ___  _ __   |_ _|__| |
##   |  _| | |/ _ \/ __| __| '__/ _ \| '_ \   | |/ _` |
##   | |___| |  __/ (__| |_| | | (_) | | | |  | | (_| |
##   |_____|_|\___|\___|\__|_|  \___/|_| |_| |___\__,_|
##   

# Electron ID  ######
process.PassingId = cms.EDFilter("GsfElectronRefSelector",
    src = cms.InputTag("gsfElectrons"),
    cut = cms.string(process.PassingIsolation.cut.value() +
          " && ( (isEB && sigmaIetaIeta<0.01 && deltaEtaSuperClusterTrackAtVtx<0.0071)"
          "|| (isEE && sigmaIetaIeta<0.028 && deltaEtaSuperClusterTrackAtVtx<0.0066) )")   
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
    InputProducer = cms.InputTag("PassingId"),                          
    hltTag = cms.untracked.InputTag("HLT_Ele15_SW_L1R","","HLT"),
    triggerEventTag = cms.untracked.InputTag("hltTriggerSummaryAOD","","HLT")
)


process.badSuperClustersClean = cms.EDFilter("CandViewCleaner",
    srcCands = cms.InputTag("goodSuperClustersClean"),
    module_label = cms.string(''),
    srcObjects = cms.VInputTag(cms.InputTag("PassingHLT")),
    deltaRMin = cms.double(0.1)
)

##    _____      _                        _  __     __             
##   | ____|_  _| |_ ___ _ __ _ __   __ _| | \ \   / /_ _ _ __ ___ 
##   |  _| \ \/ / __/ _ \ '__| '_ \ / _` | |  \ \ / / _` | '__/ __|
##   | |___ >  <| ||  __/ |  | | | | (_| | |   \ V / (_| | |  \__ \
##   |_____/_/\_\\__\___|_|  |_| |_|\__,_|_|    \_/ \__,_|_|  |___/
##   

## Here we show how to use a module to compute an external variable
#process.load("JetMETCorrections.Configuration.DefaultJEC_cff")
JET_COLL = "ak5CaloJets"
JET_CUTS = "pt > 5.0 && abs(eta)<3.0 && (.05 < emEnergyFraction < .9)"

process.superClusterDRToNearestJet = cms.EDProducer("DeltaRNearestObjectComputer",
    probes = cms.InputTag("goodSuperClusters"),
       # ^^--- NOTA BENE: if probes are defined by ref, as in this case, 
       #       this must be the full collection, not the subset by refs.
    objects = cms.InputTag(JET_COLL),
    objectSelection = cms.InputTag(JET_CUTS),
)


process.JetMultiplicityInSCEvents = cms.EDProducer("ObjectMultiplicityCounter",
    probes = cms.InputTag("goodSuperClustersClean"),
    objects = cms.InputTag(JET_COLL),
    objectSelection = cms.InputTag(JET_CUTS),
)


process.GsfDRToNearestJet = cms.EDProducer("DeltaRNearestObjectComputer",
    probes = cms.InputTag("gsfElectrons"),
    objects = cms.InputTag(JET_COLL),
    objectSelection = cms.InputTag(JET_CUTS),
)



process.JetMultiplicityInGsfEvents = cms.EDProducer("ObjectMultiplicityCounter",
    probes = cms.InputTag("gsfElectrons"),
    objects = cms.InputTag(JET_COLL),
    objectSelection = cms.InputTag(JET_CUTS),
)


process.ext_ToNearestJet_sequence = cms.Sequence(
    #process.ak5CaloL2L3 + 
    process.superClusterDRToNearestJet +
    process.JetMultiplicityInSCEvents + 
    process.GsfDRToNearestJet +
    process.JetMultiplicityInGsfEvents
    )


##    _____             ____        __ _       _ _   _             
##   |_   _|_ _  __ _  |  _ \  ___ / _(_)_ __ (_) |_(_) ___  _ __  
##     | |/ _` |/ _` | | | | |/ _ \ |_| | '_ \| | __| |/ _ \| '_ \ 
##     | | (_| | (_| | | |_| |  __/  _| | | | | | |_| | (_) | | | |
##     |_|\__,_|\__, | |____/ \___|_| |_|_| |_|_|\__|_|\___/|_| |_|
##              |___/                                              

process.Tag = process.PassingHLT.clone()

process.TagMatchedSuperClusterCandsClean = cms.EDProducer("ElectronMatchedCandidateProducer",
   src     = cms.InputTag("goodSuperClustersClean"),
   ReferenceElectronCollection = cms.untracked.InputTag("Tag"),
   deltaR =  cms.untracked.double(0.3)
)


process.ele_sequence = cms.Sequence(
    process.PassingGsf * process.GsfMatchedSuperClusterCands +
    process.PassingIsolation + process.PassingId + 
    process.PassingHLT + process.Tag*
    process.TagMatchedSuperClusterCandsClean * process.badSuperClustersClean 
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
    cut   = cms.string("60 < mass < 120"),
)


process.SCSC = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("badSuperClustersClean badSuperClustersClean"), # charge coniugate states are implied
    checkCharge = cms.bool(False),                           
    cut   = cms.string("60 < mass < 120"),
)


process.tagGsf = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("Tag PassingGsf"), # charge coniugate states are implied
    checkCharge = cms.bool(False),                                   
    cut   = cms.string("60 < mass < 120"),
)


process.tagIso = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("Tag PassingIsolation"), # charge coniugate states are implied
    checkCharge = cms.bool(False),                                   
    cut   = cms.string("60 < mass < 120"),
)


process.tagId = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("Tag PassingId"), # charge coniugate states are implied
    checkCharge = cms.bool(False),                                  
    cut   = cms.string("60 < mass < 120"),
)


process.tagHLT = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("Tag PassingHLT"), # charge coniugate states are implied
    checkCharge = cms.bool(False),                                   
    cut   = cms.string("60 < mass < 120"),
)



process.allTagsAndProbes = cms.Sequence(
    process.tagSC + process.SCSC + 
    process.tagGsf +
    process.tagIso + process.tagId + process.tagHLT
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


process.McMatchSCbad = cms.EDFilter("MCTruthDeltaRMatcherNew",
    matchPDGId = cms.vint32(11),
    src = cms.InputTag("badSuperClustersClean"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("genParticles")
)


process.McMatchGsf = cms.EDFilter("MCTruthDeltaRMatcherNew",
    matchPDGId = cms.vint32(11),
    src = cms.InputTag("PassingGsf"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("genParticles"),
    checkCharge = cms.bool(True)
)

process.McMatchIso = cms.EDFilter("MCTruthDeltaRMatcherNew",
    matchPDGId = cms.vint32(11),
    src = cms.InputTag("PassingIsolation"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("genParticles"),
    checkCharge = cms.bool(True)
)

process.McMatchId = cms.EDFilter("MCTruthDeltaRMatcherNew",
    matchPDGId = cms.vint32(11),
    src = cms.InputTag("PassingId"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("genParticles"),
    checkCharge = cms.bool(True)
)

process.McMatchHLT = cms.EDFilter("MCTruthDeltaRMatcherNew",
    matchPDGId = cms.vint32(11),
    src = cms.InputTag("PassingHLT"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("genParticles"),
    checkCharge = cms.bool(True)
)




process.mc_sequence = cms.Sequence(
   process.McMatchTag + process.McMatchSC + process.McMatchSCbad +
   process.McMatchGsf + process.McMatchIso +
   process.McMatchId  + process.McMatchHLT
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


## I define some common variables for re-use later.
## This will save us repeating the same code for each efficiency category

recoCommonStuff = cms.PSet(
    variables = cms.PSet(
        eta = cms.string("eta()"),
        pt  = cms.string("pt()"),
        phi  = cms.string("phi()"),
        et  = cms.string("et()"),
        e  = cms.string("energy()"),
        p  = cms.string("p()"),
        px  = cms.string("px()"),
        py  = cms.string("py()"),
        pz  = cms.string("pz()"),
        theta  = cms.string("theta()"),
    ),
   ignoreExceptions =  cms.bool (True),
   fillTagTree      =  cms.bool (True),  
)

mcTruthCommonStuff = cms.PSet(
   isMC = cms.bool(True),
   tagMatches = cms.InputTag("McMatchTag"),
   motherPdgId = cms.vint32(22,23),
   makeMCUnbiasTree = cms.bool(True),
   checkMotherInUnbiasEff = cms.bool(True),
)   

gsfEleCommonStuff = cms.PSet(
    variables = cms.PSet(
        eta = cms.string("eta()"),
        pt  = cms.string("pt()"),
        phi  = cms.string("phi()"),
        et  = cms.string("et()"),
        e  = cms.string("energy()"),
        p  = cms.string("p()"),
        px  = cms.string("px()"),
        py  = cms.string("py()"),
        pz  = cms.string("pz()"),
        theta  = cms.string("theta()"),    
        charge = cms.string("charge()"),
        vx     = cms.string("vx()"),
        vy     = cms.string("vy()"),
        vz     = cms.string("vz()"),
        rapidity  = cms.string("rapidity()"),
        ## isolation 
        trackiso = cms.string("dr04TkSumPt()"),
        ecaliso  = cms.string("dr04EcalRecHitSumEt()"),
        hcaliso  = cms.string("dr04HcalTowerSumEt()"),
        classification = cms.string("classification()"),
        numberOfBrems  = cms.string("numberOfBrems()"),     
        bremFraction   = cms.string("fbrem()"),
        deltaEtaIn     = cms.string("deltaEtaSuperClusterTrackAtVtx()"),
        deltaPhiIn     = cms.string("deltaPhiSuperClusterTrackAtVtx()"),
        deltaPhiOut    = cms.string("deltaPhiSeedClusterTrackAtCalo()"),
        deltaEtaOut    = cms.string("deltaEtaSeedClusterTrackAtCalo()"),
        ## Hcal energy over Ecal Energy
        HoverE = cms.string("hadronicOverEm()"),
        EoverPout = cms.string("eSeedClusterOverPout()"),
        EoverPin  = cms.string("eSuperClusterOverP()"),
        ## Cluster shape information
        sigmaEtaEta  = cms.string("sigmaEtaEta()"),
        sigmaIetaIeta = cms.string("sigmaIetaIeta()"),
        ),
    ignoreExceptions =  cms.bool (True),
    fillTagTree      =  cms.bool (True),  
)


##    ____   ____       __     ____      __ 
##   / ___| / ___|      \ \   / ___|___ / _|
##   \___ \| |      _____\ \ | |  _/ __| |_ 
##    ___) | |___  |_____/ / | |_| \__ \  _|
##   |____/ \____|      /_/   \____|___/_|  

## super cluster --> gsf electron
process.SCToGsf = cms.EDAnalyzer("TagProbeFitTreeProducer",
    ## pick the defaults
    recoCommonStuff, mcTruthCommonStuff,
    # choice of tag and probe pairs, and arbitration                 
    tagProbePairs = cms.InputTag("tagSC"),
    arbitration   = cms.string("OneProbe"),                      
    flags = cms.PSet(
        passing = cms.InputTag("GsfMatchedSuperClusterCands"),
        passingALL = cms.InputTag("TagMatchedSuperClusterCandsClean")
    ),
    probeMatches  = cms.InputTag("McMatchSC"),
    allProbes     = cms.InputTag("goodSuperClustersClean")
)
process.SCToGsf.variables.drjet = cms.InputTag("superClusterDRToNearestJet")
process.SCToGsf.variables.nJets = cms.InputTag("JetMultiplicityInSCEvents")





process.SCSCtoTagSC = cms.EDAnalyzer("TagProbeFitTreeProducer",
    ## pick the defaults
    recoCommonStuff,
    isMC = cms.bool(False),
    #mcTruthCommonStuff,
    # choice of tag and probe pairs, and arbitration                      
    tagProbePairs = cms.InputTag("SCSC"),
    arbitration   = cms.string("OneProbe"),
    massForArbitration = cms.double(91.1876),
    flags = cms.PSet(
          passing = cms.InputTag("TagMatchedSuperClusterCandsClean")
    ),
    probeMatches  = cms.InputTag("McMatchSCbad"),         
    allProbes     = cms.InputTag("badSuperClustersClean")
)


##     ____      __       __    ___           
##    / ___|___ / _|      \ \  |_ _|___  ___  
##   | |  _/ __| |_   _____\ \  | |/ __|/ _ \ 
##   | |_| \__ \  _| |_____/ /  | |\__ \ (_) |
##    \____|___/_|        /_/  |___|___/\___/ 
##   
##  gsf electron --> isolation

process.GsfToIso = cms.EDAnalyzer("TagProbeFitTreeProducer",
    mcTruthCommonStuff, gsfEleCommonStuff,                        
    tagProbePairs = cms.InputTag("tagGsf"),
    arbitration   = cms.string("OneProbe"),
    flags = cms.PSet(
        passing = cms.InputTag("PassingIsolation")
    ),
    probeMatches  = cms.InputTag("McMatchGsf"),
    allProbes     = cms.InputTag("PassingGsf")
)
process.GsfToIso.variables.drjet = cms.InputTag("GsfDRToNearestJet")
process.GsfToIso.variables.nJets = cms.InputTag("JetMultiplicityInGsfEvents")

##    ___                 __    ___    _ 
##   |_ _|___  ___        \ \  |_ _|__| |
##    | |/ __|/ _ \   _____\ \  | |/ _` |
##    | |\__ \ (_) | |_____/ /  | | (_| |
##   |___|___/\___/       /_/  |___\__,_|
##   
##  isolation --> Id

process.IsoToId = cms.EDAnalyzer("TagProbeFitTreeProducer",
    mcTruthCommonStuff, gsfEleCommonStuff,                              
    tagProbePairs = cms.InputTag("tagIso"),
    arbitration   = cms.string("OneProbe"),
    flags = cms.PSet(
        passing = cms.InputTag("PassingId")
    ),
    probeMatches  = cms.InputTag("McMatchIso"),
    allProbes     = cms.InputTag("PassingIsolation")
)
process.IsoToId.variables.drjet = cms.InputTag("GsfDRToNearestJet")
process.IsoToId.variables.nJets = cms.InputTag("JetMultiplicityInGsfEvents")

##    ___    _       __    _   _ _   _____ 
##   |_ _|__| |      \ \  | | | | | |_   _|
##    | |/ _` |  _____\ \ | |_| | |   | |  
##    | | (_| | |_____/ / |  _  | |___| |  
##   |___\__,_|      /_/  |_| |_|_____|_|  

##  Id --> HLT
process.IdToHLT = cms.EDAnalyzer("TagProbeFitTreeProducer",
    mcTruthCommonStuff, gsfEleCommonStuff,                             
    tagProbePairs = cms.InputTag("tagId"),
    arbitration   = cms.string("OneProbe"),
    flags = cms.PSet(
        passing = cms.InputTag("PassingHLT")
    ),
    probeMatches  = cms.InputTag("McMatchId"),
    allProbes     = cms.InputTag("PassingId")
)
process.IdToHLT.variables.drjet = cms.InputTag("GsfDRToNearestJet")
process.IdToHLT.variables.nJets = cms.InputTag("JetMultiplicityInGsfEvents")


process.tree_sequence = cms.Sequence(
    process.SCToGsf + process.SCSCtoTagSC +
    process.GsfToIso +
    process.IsoToId + process.IdToHLT
)    



##    ____       _   _     
##   |  _ \ __ _| |_| |__  
##   | |_) / _` | __| '_ \ 
##   |  __/ (_| | |_| | | |
##   |_|   \__,_|\__|_| |_|
##

process.tagAndProbe = cms.Path(
    process.sc_sequence + process.ele_sequence +
    process.ext_ToNearestJet_sequence + 
    process.allTagsAndProbes + process.mc_sequence + 
    process.tree_sequence
)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("testNewWrite.root")
                                   )
