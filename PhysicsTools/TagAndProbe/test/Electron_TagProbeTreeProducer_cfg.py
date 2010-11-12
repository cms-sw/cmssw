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
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

########################
MC_flag = False

HLTPath = "HLT_Ele17_SW_TightEleId_L1R"
#HLTPath = "HLT_Ele15_SW_CaloEleId_L1R"
#HLTPath = "HLT_Ele15_SW_L1R"
#HLTPath = "HLT_Ele15_LW_L1R"
#HLTPath = "HLT_Photon15_Cleaned_L1R"
########################


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



#  Photons!!! ################ 
process.goodPhotons = cms.EDFilter("PhotonSelector",
                                   src = cms.InputTag("photons"),
                                   cut = cms.string("hadronicOverEm<0.15"
                                                    " && (superCluster.energy*sin(superCluster.position.theta)>15.0)"
                                                    " && (abs(superCluster.eta)<2.5) && !(1.4442<abs(superCluster.eta)<1.566)")
                                   )


process.FilteredPhotons = cms.EDFilter("PhotonRefSelector",
         src = cms.InputTag("goodPhotons"),
         cut = cms.string(process.goodPhotons.cut.value() +
                          " && ( (isEB && (sigmaIetaIeta<0.01) && (hadronicOverEm<0.15))"
                          " || (isEE && (sigmaIetaIeta<0.03) && (hadronicOverEm<0.15)))"                
                          )
)




##   ____                         ____ _           _            
##  / ___| _   _ _ __   ___ _ __ / ___| |_   _ ___| |_ ___ _ __ 
##  \___ \| | | | '_ \ / _ \ '__| |   | | | | / __| __/ _ \ '__|
##   ___) | |_| | |_) |  __/ |  | |___| | |_| \__ \ ||  __/ |   
##  |____/ \__,_| .__/ \___|_|   \____|_|\__,_|___/\__\___|_|   
##  

#  SuperClusters  ################
process.superClusters = cms.EDProducer("SuperClusterMerger",
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
      cut = cms.string("et>15.0 && abs(eta)<2.5 && !(1.4442< abs(eta) <1.566)"),
      filter = cms.bool(True)
)                                         
                                         

#### remove real jets (with high hadronic energy fraction) from SC collection
##### this improves the purity of the probe sample without affecting efficiency

process.JetsToRemoveFromSuperCluster = cms.EDFilter("CaloJetSelector",   
    src = cms.InputTag("ak5CaloJets"),
    cut = cms.string('pt>5 && energyFractionHadronic > 0.15')
)


process.goodSuperClustersClean = cms.EDProducer("CandViewCleaner",
    srcCands = cms.InputTag("goodSuperClusters"),
    module_label = cms.string(''),
    srcObjects = cms.VInputTag(cms.InputTag("JetsToRemoveFromSuperCluster")),
    deltaRMin = cms.double(0.1)
)



## process.superClusters = cms.EDFilter("EgammaHLTRecoEcalCandidateProducers",
##    scHybridBarrelProducer =  cms.InputTag("hybridSuperClusters","", "RECO"),
##    scIslandEndcapProducer =  cms.InputTag("multi5x5SuperClustersWithPreshower","", "RECO"),    
##    recoEcalCandidateCollection = cms.string("")
## )


process.sc_sequence = cms.Sequence( process.goodPhotons *
                                    process.FilteredPhotons *
                                    process.superClusters *
                                    process.superClusterCands *
                                    process.goodSuperClusters *
                                    process.JetsToRemoveFromSuperCluster *
                                    process.goodSuperClustersClean
                                    )


##    ____      __ _____ _           _                   
##   / ___|___ / _| ____| | ___  ___| |_ _ __ ___  _ __  
##  | |  _/ __| |_|  _| | |/ _ \/ __| __| '__/ _ \| '_ \ 
##  | |_| \__ \  _| |___| |  __/ (__| |_| | | (_) | | | |
##   \____|___/_| |_____|_|\___|\___|\__|_|  \___/|_| |_|
##  

#  GsfElectron ################ 
process.PassingGsf = cms.EDFilter("GsfElectronRefSelector",
    src = cms.InputTag("gsfElectrons"),
    cut = cms.string("(abs(superCluster.eta)<2.5) && !(1.4442<abs(superCluster.eta)<1.566) && ecalDrivenSeed==1"
                     " && (ecalEnergy*sin(superClusterPosition.theta)>15.0) && (hadronicOverEm<0.15)")    
)


process.GsfMatchedSuperClusterCands = cms.EDProducer("ElectronMatchedCandidateProducer",
   src     = cms.InputTag("goodSuperClustersClean"),
   ReferenceElectronCollection = cms.untracked.InputTag("PassingGsf"),
   deltaR =  cms.untracked.double(0.3)
)


process.GsfMatchedPhotonCands = cms.EDProducer("ElectronMatchedCandidateProducer",
   src     = cms.InputTag("FilteredPhotons"),
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
         " && (( isEB && ( dr03TkSumPt/p4.Pt < 0.15 && dr03EcalRecHitSumEt/p4.Pt < 2.0 && dr03HcalTowerSumEt/p4.Pt < 0.12 ))"
         " || (isEE && (dr03TkSumPt/p4.Pt < 0.08 && dr03EcalRecHitSumEt/p4.Pt < 0.06  && dr03HcalTowerSumEt/p4.Pt < 0.05 )))")
         #" && (( isEB && ( (dr03TkSumPt + max(0., dr03EcalRecHitSumEt - 1.) + dr03HcalTowerSumEt)/(p4.Pt) < 0.15 ))"
         #" || (isEE && ((dr03TkSumPt + dr03EcalRecHitSumEt + dr03HcalTowerSumEt)/(p4.Pt) < 0.1 )))"
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
                     " && (gsfTrack.trackerExpectedHitsInner.numberOfHits <= 1)"
                     " && ((isEB"
                                   " && (sigmaIetaIeta<0.01)"
                                   " && ( -0.8<deltaPhiSuperClusterTrackAtVtx<0.8 )"
                                   " && ( -0.007<deltaEtaSuperClusterTrackAtVtx<0.007 )"
                                   " && (hadronicOverEm<0.15)"
                                   ")"
                     " || (isEE"
                                   " && (sigmaIetaIeta<0.03)"
                                   " && ( -0.7<deltaPhiSuperClusterTrackAtVtx<0.7 )"
                                   " && ( -0.01<deltaEtaSuperClusterTrackAtVtx<0.01 )"
                                   " && (hadronicOverEm<0.07) "
                                   "))"
                     ) 
)


process.PassingId80 = cms.EDFilter("GsfElectronRefSelector",
    src = cms.InputTag("gsfElectrons"),
    cut = cms.string(process.PassingGsf.cut.value() +
                     " && (gsfTrack.trackerExpectedHitsInner.numberOfHits <= 0)"
                     " && ((isEB"
                     " && ( dr03TkSumPt/p4.Pt <0.09 && dr03EcalRecHitSumEt/p4.Pt < 0.07 && dr03HcalTowerSumEt/p4.Pt  < 0.1 )"
#                                   " && ( (dr03TkSumPt + max(0., dr03EcalRecHitSumEt - 1.) + dr03HcalTowerSumEt)/(p4.Pt) < 0.07 )"
                                   " && (sigmaIetaIeta<0.01)"
                                   " && ( -0.06<deltaPhiSuperClusterTrackAtVtx<0.06 )"
                                   " && ( -0.004<deltaEtaSuperClusterTrackAtVtx<0.004 )"
                                   " && (hadronicOverEm<0.04)"
                                   ")"
                     " || (isEE"
                      " && ( dr03TkSumPt/p4.Pt <0.04 && dr03EcalRecHitSumEt/p4.Pt < 0.05 && dr03HcalTowerSumEt/p4.Pt  < 0.025 )"
#                                   " && ( (dr03TkSumPt + dr03EcalRecHitSumEt + dr03HcalTowerSumEt)/(p4.Pt) < 0.06 )"
                                   " && (sigmaIetaIeta<0.03)"
                                   " && ( -0.03<deltaPhiSuperClusterTrackAtVtx<0.03 )"
                                   " && ( -0.007<deltaEtaSuperClusterTrackAtVtx<0.007 )"
                                   " && (hadronicOverEm<0.025) "
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
process.PassingHLT = cms.EDProducer("trgMatchedGsfElectronProducer",                     
    InputProducer = cms.InputTag("PassingId"),                          
    hltTag = cms.untracked.InputTag(HLTPath,"","HLT"),
    triggerEventTag = cms.untracked.InputTag("hltTriggerSummaryAOD","","HLT")
)


process.badSuperClustersClean = cms.EDProducer("CandViewCleaner",
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
JET_CUTS = "pt > 10.0 && abs(eta)<3.0 && (0.01 < emEnergyFraction < 0.9) && (n90>5)"

process.superClusterDRToNearestJet = cms.EDProducer("DeltaRNearestObjectComputer",
    probes = cms.InputTag("goodSuperClusters"),
       # ^^--- NOTA BENE: if probes are defined by ref, as in this case, 
       #       this must be the full collection, not the subset by refs.
    objects = cms.InputTag(JET_COLL),
    objectSelection = cms.string(JET_CUTS),
)


process.JetMultiplicityInSCEvents = cms.EDProducer("CandMultiplicityCounter",
    probes = cms.InputTag("goodSuperClusters"),
    objects = cms.InputTag(JET_COLL),
    objectSelection = cms.string(JET_CUTS),
)

process.SCConvRejVars = cms.EDProducer("ElectronConversionRejectionVars",
    probes = cms.InputTag("goodSuperClusters")
)


process.PhotonConvRejVars = cms.EDProducer("ElectronConversionRejectionVars",
    probes = cms.InputTag("goodPhotons")
)



process.GsfDRToNearestJet = cms.EDProducer("DeltaRNearestObjectComputer",
    probes = cms.InputTag("gsfElectrons"),
    objects = cms.InputTag(JET_COLL),
    objectSelection = cms.string(JET_CUTS),
)



process.JetMultiplicityInGsfEvents = cms.EDProducer("CandMultiplicityCounter",
    probes = cms.InputTag("gsfElectrons"),
    objects = cms.InputTag(JET_COLL),
    objectSelection = cms.string(JET_CUTS),
)


process.GsfConvRejVars = cms.EDProducer("ElectronConversionRejectionVars",
    probes = cms.InputTag("gsfElectrons")
)
                                       

process.ext_ToNearestJet_sequence = cms.Sequence(
    #process.ak5CaloL2L3 + 
    process.superClusterDRToNearestJet +
    process.JetMultiplicityInSCEvents +
    process.SCConvRejVars +
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

process.Tag = process.PassingHLT.clone()

process.TagMatchedSuperClusterCandsClean = cms.EDProducer("ElectronMatchedCandidateProducer",
   src     = cms.InputTag("goodSuperClustersClean"),
   ReferenceElectronCollection = cms.untracked.InputTag("Tag"),
   deltaR =  cms.untracked.double(0.3)
)


process.TagMatchedPhotonCands = cms.EDProducer("ElectronMatchedCandidateProducer",
   src     = cms.InputTag("FilteredPhotons"),
   ReferenceElectronCollection = cms.untracked.InputTag("Tag"),
   deltaR =  cms.untracked.double(0.3)
)

process.IsoMatchedSuperClusterCandsClean = process.TagMatchedSuperClusterCandsClean.clone()
process.IsoMatchedSuperClusterCandsClean.ReferenceElectronCollection = cms.untracked.InputTag("PassingIsolation")
process.IdMatchedSuperClusterCandsClean = process.TagMatchedSuperClusterCandsClean.clone()
process.IdMatchedSuperClusterCandsClean.ReferenceElectronCollection = cms.untracked.InputTag("PassingId")

process.Id80MatchedSuperClusterCandsClean = process.TagMatchedSuperClusterCandsClean.clone()
process.Id80MatchedSuperClusterCandsClean.ReferenceElectronCollection = cms.untracked.InputTag("PassingId80")
process.IsoMatchedPhotonCands = process.GsfMatchedPhotonCands.clone()
process.IsoMatchedPhotonCands.ReferenceElectronCollection = cms.untracked.InputTag("PassingIsolation")

process.IdMatchedPhotonCands = process.GsfMatchedPhotonCands.clone()
process.IdMatchedPhotonCands.ReferenceElectronCollection = cms.untracked.InputTag("PassingId")

process.Id80MatchedPhotonCands = process.GsfMatchedPhotonCands.clone()
process.Id80MatchedPhotonCands.ReferenceElectronCollection = cms.untracked.InputTag("PassingId80")



process.ele_sequence = cms.Sequence(
    process.PassingGsf * process.GsfMatchedSuperClusterCands +
    process.GsfMatchedPhotonCands +
    process.PassingIsolation + process.PassingId +
    process.PassingId80 +
    process.PassingHLT + process.Tag*
    process.TagMatchedSuperClusterCandsClean *
    process.badSuperClustersClean *
    process.TagMatchedPhotonCands *
    process.IsoMatchedSuperClusterCandsClean *
    process.IdMatchedSuperClusterCandsClean *
    process.Id80MatchedSuperClusterCandsClean *
    process.IsoMatchedPhotonCands *
    process.IdMatchedPhotonCands *
    process.Id80MatchedPhotonCands    
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


process.tagPhoton = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("Tag FilteredPhotons"), # charge coniugate states are implied
    checkCharge = cms.bool(False),                           
    cut   = cms.string("40 < mass < 1000"),
)

process.SCSC = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("badSuperClustersClean badSuperClustersClean"), # charge coniugate states are implied
    checkCharge = cms.bool(False),                           
    cut   = cms.string("40 < mass < 1000"),
)

process.GsfGsf = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("PassingGsf PassingGsf"), # charge coniugate states are implied
    checkCharge = cms.bool(False),                                   
    cut   = cms.string("40 < mass < 1000"),
)

process.tagGsf = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("Tag PassingGsf"), # charge coniugate states are implied
    checkCharge = cms.bool(False),                                   
    cut   = cms.string("40 < mass < 1000"),
)


process.tagIso = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("Tag PassingIsolation"), # charge coniugate states are implied
    checkCharge = cms.bool(False),                                   
    cut   = cms.string("40 < mass < 1000"),
)


process.tagId = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("Tag PassingId"), # charge coniugate states are implied
    checkCharge = cms.bool(False),                                  
    cut   = cms.string("40 < mass < 1000"),
)


process.tagHLT = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("Tag PassingHLT"), # charge coniugate states are implied
    checkCharge = cms.bool(False),                                   
    cut   = cms.string("40 < mass < 1000"),
)


process.CSVarsTagGsf = cms.EDProducer("ColinsSoperVariablesComputer",
    parentBoson = cms.InputTag("tagGsf")
)
process.CSVarsTagIso = cms.EDProducer("ColinsSoperVariablesComputer",
    parentBoson = cms.InputTag("tagIso")
)
process.CSVarsTagId = cms.EDProducer("ColinsSoperVariablesComputer",
    parentBoson = cms.InputTag("tagId")
)

process.CSVarsGsfGsf = cms.EDProducer("ColinsSoperVariablesComputer",
    parentBoson = cms.InputTag("GsfGsf")
)


process.allTagsAndProbes = cms.Sequence(
    process.tagSC + process.SCSC + process.tagPhoton +
    process.tagGsf + process.GsfGsf +
    process.tagIso + process.tagId + process.tagHLT +
    process.CSVarsTagGsf + process.CSVarsTagIso +
    process.CSVarsTagId + process.CSVarsGsfGsf
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


process.McMatchPhoton = cms.EDFilter("MCTruthDeltaRMatcherNew",
    matchPDGId = cms.vint32(11),
    src = cms.InputTag("FilteredPhotons"),
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
   process.McMatchTag + process.McMatchSC + process.McMatchPhoton +
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
    probe_gsfEle_vx     = cms.string("vx"),
    probe_gsfEle_vy     = cms.string("vy"),
    probe_gsfEle_vz     = cms.string("vz"),
    probe_gsfEle_rapidity  = cms.string("rapidity"),
    probe_gsfEle_missingHits = cms.string("gsfTrack.trackerExpectedHitsInner.numberOfHits"),
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
    probe_sc_rawEnergy = cms.string("superCluster.rawEnergy"), 
    probe_sc_preshowerEnergy   = cms.string("superCluster.preshowerEnergy"), 
    probe_sc_phiWidth   = cms.string("superCluster.phiWidth"), 
    probe_sc_etaWidth   = cms.string("superCluster.etaWidth"),         
    ## isolation 
    probe_gsfEle_trackiso_dr04 = cms.string("dr04TkSumPt"),
    probe_gsfEle_ecaliso_dr04  = cms.string("dr04EcalRecHitSumEt"),
    probe_gsfEle_hcaliso_dr04  = cms.string("dr04HcalTowerSumEt"),
    probe_gsfEle_trackiso_dr03 = cms.string("dr03TkSumPt"),
    probe_gsfEle_ecaliso_dr03  = cms.string("dr03EcalRecHitSumEt"),
    probe_gsfEle_hcaliso_dr03  = cms.string("dr03HcalTowerSumEt"),
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
    probe_gsfEle_isEBEEGap      = cms.string("isEBEEGap"),
    probe_gsfEle_isEBGap        = cms.string("isEBGap"),
    probe_gsfEle_isEBEtaGap     = cms.string("isEBEtaGap"),
    probe_gsfEle_isEBPhiGap     = cms.string("isEBPhiGap"),
    probe_gsfEle_isEEGap        = cms.string("isEEGap"),
    probe_gsfEle_isEEDeeGap     = cms.string("isEEDeeGap"),
    probe_gsfEle_isEERingGap    = cms.string("isEERingGap"),
    ## Hcal energy over Ecal Energy
    probe_gsfEle_HoverE         = cms.string("hcalOverEcal"),
    probe_gsfEle_EoverP         = cms.string("eSuperClusterOverP"),        
    probe_gsfEle_EoverPout      = cms.string("eSeedClusterOverPout"),
    probe_gsfEle_HoverE_Depth1  = cms.string("hcalDepth1OverEcal"),
    probe_gsfEle_HoverE_Depth2  = cms.string("hcalDepth2OverEcal"),
    ## Cluster shape information
    probe_gsfEle_sigmaEtaEta  = cms.string("sigmaEtaEta"),
    probe_gsfEle_sigmaIetaIeta = cms.string("sigmaIetaIeta"),
    probe_gsfEle_e1x5               = cms.string("e1x5"),
    probe_gsfEle_e2x5Max            = cms.string("e2x5Max"),
    probe_gsfEle_e5x5               = cms.string("e5x5"),
    ## is ECAL driven ? is Track driven ?
    probe_gsfEle_ecalDrivenSeed     = cms.string("ecalDrivenSeed"),
    probe_gsfEle_trackerDrivenSeed  = cms.string("trackerDrivenSeed"),
    ## fraction of common hits between the GSF and CTF tracks
    probe_gsfEle_shFracInnerHits    = cms.string("shFracInnerHits")     
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
    gsfEle_vx     = cms.string("vx"),
    gsfEle_vy     = cms.string("vy"),
    gsfEle_vz     = cms.string("vz"),
    gsfEle_rapidity  = cms.string("rapidity"),
    gsfEle_missingHits = cms.string("gsfTrack.trackerExpectedHitsInner.numberOfHits"),
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
    sc_rawEnergy = cms.string("superCluster.rawEnergy"), 
    sc_preshowerEnergy   = cms.string("superCluster.preshowerEnergy"), 
    sc_phiWidth   = cms.string("superCluster.phiWidth"), 
    sc_etaWidth   = cms.string("superCluster.etaWidth"),         
    ## isolation 
    gsfEle_trackiso_dr04 = cms.string("dr04TkSumPt"),
    gsfEle_ecaliso_dr04  = cms.string("dr04EcalRecHitSumEt"),
    gsfEle_hcaliso_dr04  = cms.string("dr04HcalTowerSumEt"),
    gsfEle_trackiso_dr03 = cms.string("dr03TkSumPt"),
    gsfEle_ecaliso_dr03  = cms.string("dr03EcalRecHitSumEt"),
    gsfEle_hcaliso_dr03  = cms.string("dr03HcalTowerSumEt"),
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
    gsfEle_isEBEEGap      = cms.string("isEBEEGap"),
    gsfEle_isEBGap        = cms.string("isEBGap"),
    gsfEle_isEBEtaGap     = cms.string("isEBEtaGap"),
    gsfEle_isEBPhiGap     = cms.string("isEBPhiGap"),
    gsfEle_isEEGap        = cms.string("isEEGap"),
    gsfEle_isEEDeeGap     = cms.string("isEEDeeGap"),
    gsfEle_isEERingGap    = cms.string("isEERingGap"),
    ## Hcal energy over Ecal Energy
    gsfEle_HoverE         = cms.string("hcalOverEcal"),
    gsfEle_EoverP         = cms.string("eSuperClusterOverP"),        
    gsfEle_EoverPout      = cms.string("eSeedClusterOverPout"),
    gsfEle_HoverE_Depth1  = cms.string("hcalDepth1OverEcal"),
    gsfEle_HoverE_Depth2  = cms.string("hcalDepth2OverEcal"),
    ## Cluster shape information
    gsfEle_sigmaEtaEta  = cms.string("sigmaEtaEta"),
    gsfEle_sigmaIetaIeta = cms.string("sigmaIetaIeta"),
    gsfEle_e1x5               = cms.string("e1x5"),
    gsfEle_e2x5Max            = cms.string("e2x5Max"),
    gsfEle_e5x5               = cms.string("e5x5"),
    ## is ECAL driven ? is Track driven ?
    gsfEle_ecalDrivenSeed     = cms.string("ecalDrivenSeed"),
    gsfEle_trackerDrivenSeed  = cms.string("trackerDrivenSeed"),
    ## fraction of common hits between the GSF and CTF tracks
    gsfEle_shFracInnerHits    = cms.string("shFracInnerHits")
)


ProbePhotonVariablesToStore = cms.PSet(
        probe_eta = cms.string("eta"),
        probe_phi  = cms.string("phi"),
        probe_et  = cms.string("et"),
        probe_e  = cms.string("energy"),
        probe_p  = cms.string("p"),
        probe_px  = cms.string("px"),
        probe_py  = cms.string("py"),
        probe_pz  = cms.string("pz"),
        probe_theta  = cms.string("theta"),    
        ## isolation 
        probe_trkSumPtHollowConeDR03 = cms.string("trkSumPtHollowConeDR03"),
        probe_ecalRecHitSumEtConeDR03  = cms.string("ecalRecHitSumEtConeDR03"),
        probe_hcalTowerSumEtConeDR03  = cms.string("hcalTowerSumEtConeDR03"),
        ## booleans
        probe_isPhoton  = cms.string("isPhoton"),     
        probe_hasConversionTracks = cms.string("hasConversionTracks"),

        ## Hcal energy over Ecal Energy
        probe_hadronicOverEm = cms.string("hadronicOverEm"),
        ## Cluster shape information
        probe_sigmaIetaIeta = cms.string("sigmaIetaIeta"),
)


ProbeSuperClusterVariablesToStore = cms.PSet(
    probe_sc_eta = cms.string("eta"),
    probe_sc_pt  = cms.string("pt"),
    probe_sc_phi  = cms.string("phi"),
    probe_sc_et  = cms.string("et"),
    probe_sc_e  = cms.string("energy"),
    probe_sc_p  = cms.string("p"),
    probe_sc_px  = cms.string("px"),
    probe_sc_py  = cms.string("py"),
    probe_sc_pz  = cms.string("pz"),
    probe_sc_theta  = cms.string("theta"),
)


TagSuperClusterVariablesToStore = cms.PSet(
    sc_eta = cms.string("eta"),
    sc_pt  = cms.string("pt"),
    sc_phi  = cms.string("phi"),
    sc_et  = cms.string("et"),
    sc_e  = cms.string("energy"),
    sc_p  = cms.string("p"),
    sc_px  = cms.string("px"),
    sc_py  = cms.string("py"),
        sc_pz  = cms.string("pz"),
    sc_theta  = cms.string("theta"),
)





CommonStuffForSuperClusterProbe = cms.PSet(
   variables = cms.PSet(ProbeSuperClusterVariablesToStore),
   ignoreExceptions =  cms.bool (False),
   #fillTagTree      =  cms.bool (True),
   addRunLumiInfo   =  cms.bool (True),
   addEventVariablesInfo   =  cms.bool (True),
   pairVariables =  cms.PSet(ZVariablesToStore),
   pairFlags     =  cms.PSet(
          mass60to120 = cms.string("40 < mass < 1000")
    ),
    tagVariables   =  cms.PSet(TagVariablesToStore),
    tagFlags     =  cms.PSet(
          passingGsf = cms.InputTag("PassingGsf"),
          passingIso = cms.InputTag("PassingIsolation"),
          passingId = cms.InputTag("PassingId"),
          passingId80 = cms.InputTag("PassingId80"),        
          passingHLT = cms.InputTag("PassingHLT")                
    ),    
)






CommonStuffForGsfElectronProbe = cms.PSet(
    variables = cms.PSet(ProbeVariablesToStore),
    ignoreExceptions =  cms.bool (False),
    #fillTagTree      =  cms.bool (True),
    addRunLumiInfo   =  cms.bool (True),
    addEventVariablesInfo   =  cms.bool (True),
    pairVariables =  cms.PSet(ZVariablesToStore),
    pairFlags     =  cms.PSet(
          mass60to120 = cms.string("40 < mass < 1000")
    ),
    tagVariables   =  cms.PSet(TagVariablesToStore),
    tagFlags     =  cms.PSet(
          passingGsf = cms.InputTag("PassingGsf"),
          passingIso = cms.InputTag("PassingIsolation"),
          passingId = cms.InputTag("PassingId"),
          passingId80 = cms.InputTag("PassingId80"),        
          passingHLT = cms.InputTag("PassingHLT")     
    ),    
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

## super cluster --> gsf electron
process.SCToGsf = cms.EDAnalyzer("TagProbeFitTreeProducer",
    ## pick the defaults
    CommonStuffForSuperClusterProbe, mcTruthCommonStuff,
    # choice of tag and probe pairs, and arbitration                 
    tagProbePairs = cms.InputTag("tagSC"),
    arbitration   = cms.string("Random2"),                      
    flags = cms.PSet(
        probe_passing = cms.InputTag("GsfMatchedSuperClusterCands"),
        probe_passingGsf = cms.InputTag("GsfMatchedSuperClusterCands"),        
        probe_passingIso = cms.InputTag("IsoMatchedSuperClusterCandsClean"),
        probe_passingId = cms.InputTag("IdMatchedSuperClusterCandsClean"),
        probe_passingId80 = cms.InputTag("Id80MatchedSuperClusterCandsClean"),
        probe_passingALL = cms.InputTag("TagMatchedSuperClusterCandsClean")
    ),
    probeMatches  = cms.InputTag("McMatchSC"),
    allProbes     = cms.InputTag("goodSuperClustersClean")
)
process.SCToGsf.variables.probe_dRjet = cms.InputTag("superClusterDRToNearestJet")
process.SCToGsf.variables.probe_nJets = cms.InputTag("JetMultiplicityInSCEvents")
process.SCToGsf.variables.probe_dist = cms.InputTag("SCConvRejVars","dist")
process.SCToGsf.variables.probe_dcot = cms.InputTag("SCConvRejVars","dcot")
process.SCToGsf.variables.probe_convradius = cms.InputTag("SCConvRejVars","convradius")
process.SCToGsf.variables.probe_passConvRej = cms.InputTag("SCConvRejVars","passConvRej")
process.SCToGsf.tagVariables.dRjet = cms.InputTag("GsfDRToNearestJet")
process.SCToGsf.tagVariables.nJets = cms.InputTag("JetMultiplicityInGsfEvents")
process.SCToGsf.tagVariables.dist = cms.InputTag("GsfConvRejVars","dist")
process.SCToGsf.tagVariables.dcot = cms.InputTag("GsfConvRejVars","dcot")
process.SCToGsf.tagVariables.convradius = cms.InputTag("GsfConvRejVars","convradius")
process.SCToGsf.tagVariables.passConvRej = cms.InputTag("GsfConvRejVars","passConvRej")



process.SCSCtoTagSC = cms.EDAnalyzer("TagProbeFitTreeProducer",
    ## pick the defaults
   variables = cms.PSet(ProbeSuperClusterVariablesToStore),
   ignoreExceptions =  cms.bool (False),
   addRunLumiInfo   =  cms.bool (True),
   addEventVariablesInfo   =  cms.bool (True),
   pairVariables =  cms.PSet(ZVariablesToStore),
   pairFlags     =  cms.PSet(
          mass60to120 = cms.string("40 < mass < 1000")
    ),
    tagVariables   =  cms.PSet(TagSuperClusterVariablesToStore),
    tagFlags     =  cms.PSet(
          flag = cms.string("pt>0")
    ),                                         
    isMC = cms.bool(False),
    #mcTruthCommonStuff,
    # choice of tag and probe pairs, and arbitration                      
    tagProbePairs = cms.InputTag("SCSC"),
    arbitration   = cms.string("Random2"),
    massForArbitration = cms.double(91.1876),
    flags = cms.PSet(
          probe_passing = cms.InputTag("TagMatchedSuperClusterCandsClean")
    ),
    probeMatches  = cms.InputTag("McMatchSCbad"),         
    allProbes     = cms.InputTag("badSuperClustersClean")
)


## good photon --> gsf electron
process.PhotonToGsf = cms.EDAnalyzer("TagProbeFitTreeProducer",
    ## pick the defaults
    mcTruthCommonStuff,
    CommonStuffForSuperClusterProbe,
    # choice of tag and probe pairs, and arbitration                 
    tagProbePairs = cms.InputTag("tagPhoton"),
    arbitration   = cms.string("Random2"),                      
    flags = cms.PSet(
        probe_passing = cms.InputTag("GsfMatchedPhotonCands"),
        probe_passingALL = cms.InputTag("TagMatchedPhotonCands"),
        probe_passingIso = cms.InputTag("IsoMatchedPhotonCands"),
        probe_passingId = cms.InputTag("IdMatchedPhotonCands"),
        probe_passingId80 = cms.InputTag("Id80MatchedPhotonCands")        
    ),
    probeMatches  = cms.InputTag("McMatchPhoton"),
    allProbes     = cms.InputTag("FilteredPhotons")
)
process.PhotonToGsf.variables=ProbePhotonVariablesToStore
process.PhotonToGsf.variables.probe_dist = cms.InputTag("PhotonConvRejVars","dist")
process.PhotonToGsf.variables.probe_dcot = cms.InputTag("PhotonConvRejVars","dcot")
process.PhotonToGsf.variables.probe_convradius = cms.InputTag("PhotonConvRejVars","convradius")
process.PhotonToGsf.variables.probe_passConvRej = cms.InputTag("PhotonConvRejVars","passConvRej")
process.PhotonToGsf.tagVariables.dRjet = cms.InputTag("GsfDRToNearestJet")
process.PhotonToGsf.tagVariables.nJets = cms.InputTag("JetMultiplicityInGsfEvents")
process.PhotonToGsf.tagVariables.dist = cms.InputTag("GsfConvRejVars","dist")
process.PhotonToGsf.tagVariables.dcot = cms.InputTag("GsfConvRejVars","dcot")
process.PhotonToGsf.tagVariables.convradius = cms.InputTag("GsfConvRejVars","convradius")
process.PhotonToGsf.tagVariables.passConvRej = cms.InputTag("GsfConvRejVars","passConvRej")


process.SCSCbad = cms.EDAnalyzer("TagProbeFitTreeProducer",
    ## pick the defaults
   #######mcTruthCommonStuff,
   variables = cms.PSet(ProbeSuperClusterVariablesToStore),
   ignoreExceptions =  cms.bool (False),
   addRunLumiInfo   =  cms.bool (True),
   addEventVariablesInfo   =  cms.bool (True),
   pairVariables =  cms.PSet(ZVariablesToStore),
   pairFlags     =  cms.PSet(
          mass60to120 = cms.string("40 < mass < 1000")
          ),
   tagVariables   =  cms.PSet(TagSuperClusterVariablesToStore),
   tagFlags     =  cms.PSet(
          flag = cms.string("pt>0")
   ),                                         
   isMC = cms.bool(False),
   # choice of tag and probe pairs, and arbitration                      
   tagProbePairs = cms.InputTag("SCSC"),
   arbitration   = cms.string("Random2"),
   massForArbitration = cms.double(91.1876),
   flags = cms.PSet(
          probe_passing = cms.InputTag("TagMatchedSuperClusterCandsClean")
   ),
   #probeMatches  = cms.InputTag("McMatchSCbad"),         
   allProbes     = cms.InputTag("badSuperClustersClean")
)

process.GsfGsfToIso = cms.EDAnalyzer("TagProbeFitTreeProducer",
    ########mcTruthCommonStuff,
    CommonStuffForGsfElectronProbe,
    isMC = cms.bool(False), 
    tagProbePairs = cms.InputTag("GsfGsf"),
    arbitration   = cms.string("Random2"),
    flags = cms.PSet(
        probe_passing = cms.InputTag("PassingIsolation")
    ),
    #probeMatches  = cms.InputTag("McMatchGsf"),
    allProbes     = cms.InputTag("PassingGsf")
)
process.GsfGsfToIso.variables.probe_dRjet = cms.InputTag("GsfDRToNearestJet")
process.GsfGsfToIso.variables.probe_nJets = cms.InputTag("JetMultiplicityInGsfEvents")
process.GsfGsfToIso.variables.probe_dist = cms.InputTag("GsfConvRejVars","dist")
process.GsfGsfToIso.variables.probe_dcot = cms.InputTag("GsfConvRejVars","dcot")
process.GsfGsfToIso.variables.probe_convradius = cms.InputTag("GsfConvRejVars","convradius")
process.GsfGsfToIso.variables.probe_passConvRej = cms.InputTag("GsfConvRejVars","passConvRej")
process.GsfGsfToIso.tagVariables.dRjet = cms.InputTag("GsfDRToNearestJet")
process.GsfGsfToIso.tagVariables.nJets = cms.InputTag("JetMultiplicityInGsfEvents")
process.GsfGsfToIso.tagVariables.dist = cms.InputTag("GsfConvRejVars","dist")
process.GsfGsfToIso.tagVariables.dcot = cms.InputTag("GsfConvRejVars","dcot")
process.GsfGsfToIso.tagVariables.convradius = cms.InputTag("GsfConvRejVars","convradius")
process.GsfGsfToIso.tagVariables.passConvRej = cms.InputTag("GsfConvRejVars","passConvRej")
process.GsfGsfToIso.pairVariables.costheta = cms.InputTag("CSVarsGsfGsf","costheta")
process.GsfGsfToIso.pairVariables.sin2theta = cms.InputTag("CSVarsGsfGsf","sin2theta")
process.GsfGsfToIso.pairVariables.tanphi = cms.InputTag("CSVarsGsfGsf","tanphi")


##     ____      __       __    ___           
##    / ___|___ / _|      \ \  |_ _|___  ___  
##   | |  _/ __| |_   _____\ \  | |/ __|/ _ \ 
##   | |_| \__ \  _| |_____/ /  | |\__ \ (_) |
##    \____|___/_|        /_/  |___|___/\___/ 
##   
##  gsf electron --> isolation

process.GsfToIso = cms.EDAnalyzer("TagProbeFitTreeProducer",
    mcTruthCommonStuff, CommonStuffForGsfElectronProbe,                        
    tagProbePairs = cms.InputTag("tagGsf"),
    arbitration   = cms.string("Random2"),
    flags = cms.PSet(
        probe_passing = cms.InputTag("PassingIsolation"),
        probe_passingIso = cms.InputTag("PassingIsolation"),
        probe_passingId = cms.InputTag("PassingId"),
        probe_passingId80 = cms.InputTag("PassingId80"),        
        probe_passingALL = cms.InputTag("PassingHLT")        
    ),
    probeMatches  = cms.InputTag("McMatchGsf"),
    allProbes     = cms.InputTag("PassingGsf")
)
process.GsfToIso.variables.probe_dRjet = cms.InputTag("GsfDRToNearestJet")
process.GsfToIso.variables.probe_nJets = cms.InputTag("JetMultiplicityInGsfEvents")
process.GsfToIso.variables.probe_dist = cms.InputTag("GsfConvRejVars","dist")
process.GsfToIso.variables.probe_dcot = cms.InputTag("GsfConvRejVars","dcot")
process.GsfToIso.variables.probe_convradius = cms.InputTag("GsfConvRejVars","convradius")
process.GsfToIso.variables.probe_passConvRej = cms.InputTag("GsfConvRejVars","passConvRej")
process.GsfToIso.tagVariables.dRjet = cms.InputTag("GsfDRToNearestJet")
process.GsfToIso.tagVariables.nJets = cms.InputTag("JetMultiplicityInGsfEvents")
process.GsfToIso.tagVariables.dist = cms.InputTag("GsfConvRejVars","dist")
process.GsfToIso.tagVariables.dcot = cms.InputTag("GsfConvRejVars","dcot")
process.GsfToIso.tagVariables.convradius = cms.InputTag("GsfConvRejVars","convradius")
process.GsfToIso.tagVariables.passConvRej = cms.InputTag("GsfConvRejVars","passConvRej")
process.GsfToIso.pairVariables.costheta = cms.InputTag("CSVarsTagGsf","costheta")
process.GsfToIso.pairVariables.sin2theta = cms.InputTag("CSVarsTagGsf","sin2theta")
process.GsfToIso.pairVariables.tanphi = cms.InputTag("CSVarsTagGsf","tanphi")


##    ___                 __    ___    _ 
##   |_ _|___  ___        \ \  |_ _|__| |
##    | |/ __|/ _ \   _____\ \  | |/ _` |
##    | |\__ \ (_) | |_____/ /  | | (_| |
##   |___|___/\___/       /_/  |___\__,_|
##   
##  isolation --> Id

process.IsoToId = cms.EDAnalyzer("TagProbeFitTreeProducer",
    mcTruthCommonStuff, CommonStuffForGsfElectronProbe,                              
    tagProbePairs = cms.InputTag("tagIso"),
    arbitration   = cms.string("Random2"),
    flags = cms.PSet(
        probe_passing = cms.InputTag("PassingId"),
        probe_passingId = cms.InputTag("PassingId"),
        probe_passingId80 = cms.InputTag("PassingId80"),        
        probe_passingALL = cms.InputTag("PassingHLT")         
    ),
    probeMatches  = cms.InputTag("McMatchIso"),
    allProbes     = cms.InputTag("PassingIsolation")
)
process.IsoToId.variables.probe_dRjet = cms.InputTag("GsfDRToNearestJet")
process.IsoToId.variables.probe_nJets = cms.InputTag("JetMultiplicityInGsfEvents")
process.IsoToId.variables.probe_dist = cms.InputTag("GsfConvRejVars","dist")
process.IsoToId.variables.probe_dcot = cms.InputTag("GsfConvRejVars","dcot")
process.IsoToId.variables.probe_convradius = cms.InputTag("GsfConvRejVars","convradius")
process.IsoToId.variables.probe_passConvRej = cms.InputTag("GsfConvRejVars","passConvRej")
process.IsoToId.tagVariables.dRjet = cms.InputTag("GsfDRToNearestJet")
process.IsoToId.tagVariables.nJets = cms.InputTag("JetMultiplicityInGsfEvents")
process.IsoToId.tagVariables.dist = cms.InputTag("GsfConvRejVars","dist")
process.IsoToId.tagVariables.dcot = cms.InputTag("GsfConvRejVars","dcot")
process.IsoToId.tagVariables.convradius = cms.InputTag("GsfConvRejVars","convradius")
process.IsoToId.tagVariables.passConvRej = cms.InputTag("GsfConvRejVars","passConvRej")
process.IsoToId.pairVariables.costheta = cms.InputTag("CSVarsTagIso","costheta")
process.IsoToId.pairVariables.sin2theta = cms.InputTag("CSVarsTagIso","sin2theta")
process.IsoToId.pairVariables.tanphi = cms.InputTag("CSVarsTagIso","tanphi")



##    ___    _       __    _   _ _   _____ 
##   |_ _|__| |      \ \  | | | | | |_   _|
##    | |/ _` |  _____\ \ | |_| | |   | |  
##    | | (_| | |_____/ / |  _  | |___| |  
##   |___\__,_|      /_/  |_| |_|_____|_|  

##  Id --> HLT
process.IdToHLT = cms.EDAnalyzer("TagProbeFitTreeProducer",
    mcTruthCommonStuff, CommonStuffForGsfElectronProbe,                             
    tagProbePairs = cms.InputTag("tagId"),
    arbitration   = cms.string("Random2"),
    flags = cms.PSet(
        probe_passing = cms.InputTag("PassingHLT"),
        probe_passingId80 = cms.InputTag("PassingId80")        
    ),
    probeMatches  = cms.InputTag("McMatchId"),
    allProbes     = cms.InputTag("PassingId")
)
process.IdToHLT.variables.probe_dRjet = cms.InputTag("GsfDRToNearestJet")
process.IdToHLT.variables.probe_nJets = cms.InputTag("JetMultiplicityInGsfEvents")
process.IdToHLT.variables.probe_dist = cms.InputTag("GsfConvRejVars","dist")
process.IdToHLT.variables.probe_dcot = cms.InputTag("GsfConvRejVars","dcot")
process.IdToHLT.variables.probe_convradius = cms.InputTag("GsfConvRejVars","convradius")
process.IdToHLT.variables.probe_passConvRej = cms.InputTag("GsfConvRejVars","passConvRej")
process.IdToHLT.tagVariables.dRjet = cms.InputTag("GsfDRToNearestJet")
process.IdToHLT.tagVariables.nJets = cms.InputTag("JetMultiplicityInGsfEvents")
process.IdToHLT.tagVariables.dist = cms.InputTag("GsfConvRejVars","dist")
process.IdToHLT.tagVariables.dcot = cms.InputTag("GsfConvRejVars","dcot")
process.IdToHLT.tagVariables.convradius = cms.InputTag("GsfConvRejVars","convradius")
process.IdToHLT.tagVariables.passConvRej = cms.InputTag("GsfConvRejVars","passConvRej")
process.IdToHLT.pairVariables.costheta = cms.InputTag("CSVarsTagId","costheta")
process.IdToHLT.pairVariables.sin2theta = cms.InputTag("CSVarsTagId","sin2theta")
process.IdToHLT.pairVariables.tanphi = cms.InputTag("CSVarsTagId","tanphi")


process.tree_sequence = cms.Sequence(
    process.SCToGsf +
    process.SCSCbad +
    process.PhotonToGsf +
    process.GsfToIso +
    process.GsfGsfToIso +
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
    process.allTagsAndProbes +
    #process.mc_sequence + 
    process.tree_sequence
)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("testNewWrite.root")
                                   )
