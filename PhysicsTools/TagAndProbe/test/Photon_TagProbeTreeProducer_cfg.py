import FWCore.ParameterSet.Config as cms

##                      _              _       
##   ___ ___  _ __  ___| |_ __ _ _ __ | |_ ___ 
##  / __/ _ \| '_ \/ __| __/ _` | '_ \| __/ __|
## | (_| (_) | | | \__ \ || (_| | | | | |_\__ \
##  \___\___/|_| |_|___/\__\__,_|_| |_|\__|___/
##                                              
########################
MC_flag = False
#GLOBAL_TAG = "GR_R_38X_V13::All"
GLOBAL_TAG = 'GR_R_39X_V4::All'
OUTPUT_FILE_NAME = "Photon_tagProbeTree.root"
HLTPath1 = "HLT_Photon50_Cleaned_L1R_v1"
HLTPath2 = "HLT_Photon30_Cleaned_L1R_v1"
HLTPath3 = "HLT_Photon20_Cleaned_L1R"
#InputTagProcess = "REDIGI36X"
InputTagProcess = "HLT"
RECOProcess = "RECO"
JET_COLL = "ak5PFJets"
JET_CUTS = "abs(eta)<2.6 && chargedHadronEnergyFraction>0 && electronEnergyFraction<0.1 && nConstituents>1 && neutralHadronEnergyFraction<0.99 && neutralEmEnergyFraction<0.99 && pt>15.0" 
ELECTRON_ET_CUT_MIN = 20.0
ELECTRON_COLL = "gsfElectrons"
ELECTRON_CUTS = "ecalDrivenSeed==1 && (abs(superCluster.eta)<2.5) && !(1.4442<abs(superCluster.eta)<1.566) && (ecalEnergy*sin(superClusterPosition.theta)>" + str(ELECTRON_ET_CUT_MIN) + ")"

##    ___            _           _      
##   |_ _|_ __   ___| |_   _  __| | ___ 
##    | || '_ \ / __| | | | |/ _` |/ _ \
##    | || | | | (__| | |_| | (_| |  __/
##   |___|_| |_|\___|_|\__,_|\__,_|\___|

process = cms.Process("TagProbe")
#stuff needed for prescales
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = GLOBAL_TAG
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
##process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True),
##                                     SkipEvent = cms.untracked.vstring('ProductNotFound')
##                                     )
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

##   ____             _ ____                           
##  |  _ \ ___   ___ | / ___|  ___  _   _ _ __ ___ ___ 
##  | |_) / _ \ / _ \| \___ \ / _ \| | | | '__/ __/ _ \
##  |  __/ (_) | (_) | |___) | (_) | |_| | | | (_|  __/
##  |_|   \___/ \___/|_|____/ \___/ \__,_|_|  \___\___|
##  

readFiles = cms.untracked.vstring()

process.source = cms.Source("PoolSource", 
                            fileNames = readFiles
                            )
readFiles.extend([
    #FILES
    ])

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )    
process.source.inputCommands = cms.untracked.vstring("keep *","drop *_MEtoEDMConverter_*_*")



##  ____  _           _                                _               
## |  _ \| |__   ___ | |_ ___  _ __    _ __  _ __ ___ | |__   ___  ___ 
## | |_) | '_ \ / _ \| __/ _ \| '_ \  | '_ \| '__/ _ \| '_ \ / _ \/ __|
## |  __/| | | | (_) | || (_) | | | | | |_) | | | (_) | |_) |  __/\__ \
## |_|   |_| |_|\___/ \__\___/|_| |_| | .__/|_|  \___/|_.__/ \___||___/
##                                    |_|                              

#basic probe photon selection
#keep EB and EE efficiencies separate
#loose track match requirement cuts down on non-Z-electron background
process.probePhotons = cms.EDProducer("TrackMatchedPhotonProducer",
    srcObject = cms.InputTag("photons", "", RECOProcess),
    srcObjectsToMatch = cms.VInputTag(cms.InputTag("generalTracks")),
    srcObjectSelection = cms.string("et>20.0 && abs(eta)<2.5"),
    srcObjectsToMatchSelection = cms.string('pt > 20.0 && quality("highPurity")'),  
    deltaRMax = cms.double(0.3)
    )

##     ___           _       _   _             
##    |_ _|___  ___ | | __ _| |_(_) ___  _ __  
##     | |/ __|/ _ \| |/ _` | __| |/ _ \| '_ \ 
##     | |\__ \ (_) | | (_| | |_| | (_) | | | |
##    |___|___/\___/|_|\__,_|\__|_|\___/|_| |_|

                                         
#  Isolation ################
#ECAL and HCAL only
process.photonIsolation = cms.EDFilter("PhotonRefSelector",
    src = cms.InputTag("probePhotons"),
    cut = cms.string(
    "(ecalRecHitSumEtConeDR04 < (0.006*pt + 4.2))"
    " && (hcalTowerSumEtConeDR04 < (0.0025*pt + 2.2 ))"
    )
)

##  ____  _           _                ___    _ 
## |  _ \| |__   ___ | |_ ___  _ __   |_ _|__| |
## | |_) | '_ \ / _ \| __/ _ \| '_ \   | |/ _` |
## |  __/| | | | (_) | || (_) | | | |  | | (_| |
## |_|   |_| |_|\___/ \__\___/|_| |_| |___\__,_|
##        
#track isolation
process.photonIDsusydiphoton = process.photonIsolation.clone()
process.photonIDsusydiphoton.cut = cms.string(
    "(hadronicOverEm < 0.05) && (trkSumPtHollowConeDR04 < (0.001*pt + 2.0)"
    " && (sigmaIetaIeta < 0.013))"
    )

process.photonIDiso = process.photonIsolation.clone()
process.photonIDiso.cut = cms.string(
    "hadronicOverEm < 0.05 && ecalRecHitSumEtConeDR04 < 2.4"
    "&& hcalTowerSumEtConeDR04 < 1.0 && trkSumPtHollowConeDR04 < 0.9"
    " &&sigmaIetaIeta > 0.001"
    )

process.photonIDreliso = process.photonIsolation.clone()
process.photonIDreliso.cut = cms.string(
    "hadronicOverEm < 0.05 && ecalRecHitSumEtConeDR04 < 4.2+0.001*pt"
    " && hcalTowerSumEtConeDR04 < 2.2+0.003*pt"
    " && trkSumPtHollowConeDR04 < 2+0.001*pt && sigmaIetaIeta > 0.001"
    )

process.photonIDreliso2 = process.photonIsolation.clone()
process.photonIDreliso2.cut = cms.string(
    "hadronicOverEm < 0.05 && ecalRecHitSumEtConeDR04 < 4.2+0.001*pt"
    " && hcalTowerSumEtConeDR04 < 2.2+0.003*pt && sigmaIetaIeta > 0.001"
    )


##    _____     _                         __  __       _       _     _             
##   |_   _| __(_) __ _  __ _  ___ _ __  |  \/  | __ _| |_ ___| |__ (_)_ __   __ _ 
##     | || '__| |/ _` |/ _` |/ _ \ '__| | |\/| |/ _` | __/ __| '_ \| | '_ \ / _` |
##     | || |  | | (_| | (_| |  __/ |    | |  | | (_| | || (__| | | | | | | | (_| |
##     |_||_|  |_|\__, |\__, |\___|_|    |_|  |_|\__,_|\__\___|_| |_|_|_| |_|\__, |
##                |___/ |___/                                                |___/ 
##   
process.probePhotonsPassingHLT = cms.EDProducer(
    "trgMatchedPhotonProducer",                     
    InputProducer = cms.InputTag("probePhotons"),
    hltTags = cms.VInputTag(
    cms.InputTag(HLTPath1,"",InputTagProcess),
    cms.InputTag(HLTPath2,"",InputTagProcess),
    cms.InputTag(HLTPath3,"",InputTagProcess),
    ),
    triggerEventTag = cms.untracked.InputTag("hltTriggerSummaryAOD","",InputTagProcess),
    triggerResultsTag = cms.untracked.InputTag("TriggerResults", "", InputTagProcess)
    )


##    _____      _                        _  __     __             
##   | ____|_  _| |_ ___ _ __ _ __   __ _| | \ \   / /_ _ _ __ ___ 
##   |  _| \ \/ / __/ _ \ '__| '_ \ / _` | |  \ \ / / _` | '__/ __|
##   | |___ >  <| ||  __/ |  | | | | (_| | |   \ V / (_| | |  \__ \
##   |_____/_/\_\\__\___|_|  |_| |_|\__,_|_|    \_/ \__,_|_|  |___/
##   

## Here we show how to use a module to compute an external variable
#producer of dR < 0.5 photon-cleaned jets
process.cleanJets = cms.EDProducer("JetViewCleaner",
    srcObject = cms.InputTag(JET_COLL, "", "RECO"),
    srcObjectSelection = cms.string(JET_CUTS),
    srcObjectsToRemove = cms.VInputTag( cms.InputTag("photons", "", RECOProcess)),
    deltaRMin = cms.double(0.5)  
    )


#produce dR(photon, nearest IDed uncorrected jet passing cuts on corrected eta and pT)
process.photonDRToNearestJet = cms.EDProducer("DeltaRNearestJetComputer",
    probes = cms.InputTag("probePhotons"),
       # ^^--- NOTA BENE: if probes are defined by ref, as in this case, 
       #       this must be the full collection, not the subset by refs.
    objects = cms.InputTag("cleanJets"),
    objectSelection = cms.string(JET_CUTS)
)


#count jets passing cuts
process.JetMultiplicity = cms.EDProducer("CandMultiplicityCounter",
    probes = cms.InputTag("probePhotons"),
    objects = cms.InputTag("cleanJets"),
    objectSelection = cms.string(JET_CUTS),
    )


process.ext_ToNearestJet_sequence = cms.Sequence(
    process.cleanJets + 
    process.photonDRToNearestJet +
    process.JetMultiplicity
    )


##    _____             ____        __ _       _ _   _             
##   |_   _|_ _  __ _  |  _ \  ___ / _(_)_ __ (_) |_(_) ___  _ __  
##     | |/ _` |/ _` | | | | |/ _ \ |_| | '_ \| | __| |/ _ \| '_ \ 
##     | | (_| | (_| | | |_| |  __/  _| | | | | | |_| | (_) | | | |
##     |_|\__,_|\__, | |____/ \___|_| |_|_| |_|_|\__|_|\___/|_| |_|
##              |___/                                              

## tag should be a well reconstructed electron. We use VBTF WP80.
process.ElectronPassingWP80 = cms.EDFilter("GsfElectronRefSelector",
    src = cms.InputTag( ELECTRON_COLL ),
    cut = cms.string( ELECTRON_CUTS  +
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
)

process.Tag = process.ElectronPassingWP80.clone()
process.photon_sequence = cms.Sequence(
    process.probePhotons +
    process.photonIsolation +
    process.photonIDsusydiphoton +
    process.photonIDiso +
    process.photonIDreliso +
    process.photonIDreliso2 +
    process.probePhotonsPassingHLT + 
    process.ElectronPassingWP80 +
    process.Tag
    )


##    _____ ___   ____    ____       _          
##   |_   _( _ ) |  _ \  |  _ \ __ _(_)_ __ ___ 
##     | | / _ \/\ |_) | | |_) / _` | | '__/ __|
##     | || (_>  <  __/  |  __/ (_| | | |  \__ \
##     |_| \___/\/_|     |_|   \__,_|_|_|  |___/
##                                              
##   
#  Tag & probe selection ######
process.tagPhoton = cms.EDProducer("CandViewShallowCloneCombiner",
                                   decay = cms.string("Tag probePhotons"),
                                   checkCharge = cms.bool(False),
                                   cut = cms.string("60 < mass < 120")
                                   )
process.tagphotonIDsusydiphoton = process.tagPhoton.clone()
process.tagphotonIDsusydiphoton.decay = cms.string("Tag photonIDsusydiphoton")
process.tagphotonIDiso = process.tagPhoton.clone()
process.tagphotonIDiso.decay = cms.string("Tag photonIDiso")
process.tagphotonIDreliso = process.tagPhoton.clone()
process.tagphotonIDreliso.decay = cms.string("Tag photonIDreliso")
process.tagphotonIDreliso2 = process.tagPhoton.clone()
process.tagphotonIDreliso2.decay = cms.string("Tag photonIDreliso2")

process.allTagsAndProbes = cms.Sequence(
    process.tagPhoton +
    process.tagphotonIDsusydiphoton +
    process.tagphotonIDiso +
    process.tagphotonIDreliso +
    process.tagphotonIDreliso2 
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
process.McMatchPhoton = process.McMatchTag.clone()
process.McMatchPhoton.src = cms.InputTag("probePhotons")
process.McMatchId_susy_diphoton = process.McMatchTag.clone()
process.McMatchId_susy_diphoton.src = cms.InputTag("photonIDsusydiphoton")
process.McMatchId_iso = process.McMatchTag.clone()
process.McMatchId_iso.src = cms.InputTag("photonIDiso")
process.McMatchId_rel_iso = process.McMatchTag.clone()
process.McMatchId_rel_iso.src = cms.InputTag("photonIDreliso")
process.McMatchId_rel_iso_2 = process.McMatchTag.clone()
process.McMatchId_rel_iso_2.src = cms.InputTag("photonIDreliso2")
process.McMatchHLT = process.McMatchTag.clone()
process.McMatchHLT.src = cms.InputTag("probePhotonsPassingHLT")

process.mc_sequence = cms.Sequence(
   process.McMatchTag +
   process.McMatchPhoton +
   process.McMatchId_susy_diphoton  +
   process.McMatchId_iso +
   process.McMatchId_rel_iso +
   process.McMatchId_rel_iso_2 +
   process.McMatchHLT
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


TagPhotonVariablesToStore = cms.PSet(
    eta = cms.string("eta"),
    pt  = cms.string("pt"),
    phi  = cms.string("phi"),
    px  = cms.string("px"),
    py  = cms.string("py"),
    pz  = cms.string("pz"),
    ## super cluster quantities
    sc_energy = cms.string("superCluster.energy"),
    sc_et     = cms.string("superCluster.energy*sin(superCluster.position.theta)"),    
    sc_eta    = cms.string("superCluster.eta"),
    sc_phi    = cms.string("superCluster.phi"),
)


ProbePhotonVariablesToStore = cms.PSet(
        probe_eta = cms.string("eta"),
        probe_phi  = cms.string("phi"),
        probe_et  = cms.string("et"),
        probe_px  = cms.string("px"),
        probe_py  = cms.string("py"),
        probe_pz  = cms.string("pz"),
        ## isolation 
        probe_trkSumPtHollowConeDR03 = cms.string("trkSumPtHollowConeDR03"),
        probe_ecalRecHitSumEtConeDR03  = cms.string("ecalRecHitSumEtConeDR03"),
        probe_hcalTowerSumEtConeDR03  = cms.string("hcalTowerSumEtConeDR03"),
        probe_trkSumPtHollowConeDR04 = cms.string("trkSumPtHollowConeDR04"),
        probe_ecalRecHitSumEtConeDR04  = cms.string("ecalRecHitSumEtConeDR04"),
        probe_hcalTowerSumEtConeDR04  = cms.string("hcalTowerSumEtConeDR04"),
        ## booleans
        probe_isPhoton  = cms.string("isPhoton"),     

        ## Hcal energy over Ecal Energy
        probe_hadronicOverEm = cms.string("hadronicOverEm"),
        ## Cluster shape information
        probe_sigmaIetaIeta = cms.string("sigmaIetaIeta"),
        ## Pixel seed
        probe_hasPixelSeed = cms.string("hasPixelSeed")
)


CommonStuffForPhotonProbe = cms.PSet(
   variables = cms.PSet(ProbePhotonVariablesToStore),
   ignoreExceptions =  cms.bool (False),
   #fillTagTree      =  cms.bool (True),
   addRunLumiInfo   =  cms.bool (True),
   addEventVariablesInfo   =  cms.bool (True),
   pairVariables =  cms.PSet(ZVariablesToStore),
   pairFlags     =  cms.PSet(
          mass60to120 = cms.string("60 < mass < 120")
    ),
    tagVariables   =  cms.PSet(TagPhotonVariablesToStore),
    tagFlags     =  cms.PSet(
          flag = cms.string("pt>0")
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


##    ___                 ___    _ 
##  |_ _|___  ___       |_ _|__| |
##   | |/ __|/ _ \       | |/ _` |
##   | |\__ \ (_) |  _   | | (_| |
##   |___|___/\___/  ( ) |___\__,_|
##                   |/            
##  Photon --> isolation, id  etc.

process.PhotonToIsoId = cms.EDAnalyzer("TagProbeFitTreeProducer",
    ## pick the defaults
    mcTruthCommonStuff,
    CommonStuffForPhotonProbe,
    # choice of tag and probe pairs, and arbitration                 
    tagProbePairs = cms.InputTag("tagPhoton"),
    arbitration   = cms.string("None"),                      
    flags = cms.PSet(
        probe_passingIso = cms.InputTag("photonIsolation"),
        probe_passingHLT = cms.InputTag("probePhotonsPassingHLT"),
        probe_passingId_susy_diphoton = cms.InputTag("photonIDsusydiphoton"),
        probe_passingId_iso = cms.InputTag("photonIDiso"),
        probe_passingId_rel_iso = cms.InputTag("photonIDreliso"),
        probe_passingId_rel_iso_2 = cms.InputTag("photonIDreliso2"),
    ),
    probeMatches  = cms.InputTag("McMatchPhoton"),
    allProbes     = cms.InputTag("probePhotons")
)
process.PhotonToIsoId.variables.probe_dRjet = cms.InputTag("photonDRToNearestJet")
process.PhotonToIsoId.variables.probe_nJets = cms.InputTag("JetMultiplicity")



##    ___    _       __    _   _ _   _____ 
##   |_ _|__| |      \ \  | | | | | |_   _|
##    | |/ _` |  _____\ \ | |_| | |   | |  
##    | | (_| | |_____/ / |  _  | |___| |  
##   |___\__,_|      /_/  |_| |_|_____|_|  

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


process.photonIDsusydiphotonToHLT = cms.EDAnalyzer("TagProbeFitTreeProducer",
    HLTmcTruthCommonStuff,                                
    variables = cms.PSet(
      probe_eta = cms.string("eta"),
      probe_phi  = cms.string("phi"),
      probe_et  = cms.string("et"),
    ),
    ignoreExceptions =  cms.bool (False),
    addRunLumiInfo   =  cms.bool (False),
    addEventVariablesInfo   =  cms.bool (False),                                                        
    tagProbePairs = cms.InputTag("tagphotonIDsusydiphoton"),
    arbitration   = cms.string("None"),
    flags = cms.PSet( 
        probe_passingHLT = cms.InputTag("probePhotonsPassingHLT")        
    ),
    probeMatches  = cms.InputTag("McMatchId_susy_diphoton"),
    allProbes     = cms.InputTag("photonIDsusydiphoton")
)

process.photonIDisoToHLT = process.photonIDsusydiphotonToHLT.clone()
process.photonIDisoToHLT.tagProbePairs = cms.InputTag("tagphotonIDiso")
process.photonIDisoToHLT.probeMatches  = cms.InputTag("McMatchId_iso")
process.photonIDisoToHLT.allProbes     = cms.InputTag("photonIDiso")

process.photonIDrelisoToHLT = process.photonIDsusydiphotonToHLT.clone()
process.photonIDrelisoToHLT.tagProbePairs = cms.InputTag("tagphotonIDreliso")
process.photonIDrelisoToHLT.probeMatches  = cms.InputTag("McMatchId_rel_iso")
process.photonIDrelisoToHLT.allProbes     = cms.InputTag("photonIDreliso")

process.photonIDreliso2ToHLT = process.photonIDsusydiphotonToHLT.clone()
process.photonIDreliso2ToHLT.tagProbePairs = cms.InputTag("tagphotonIDreliso2")
process.photonIDreliso2ToHLT.probeMatches  = cms.InputTag("McMatchId_rel_iso_2")
process.photonIDreliso2ToHLT.allProbes     = cms.InputTag("photonIDreliso2")



process.tree_sequence = cms.Sequence(
    process.PhotonToIsoId +
    process.photonIDsusydiphotonToHLT +
    process.photonIDisoToHLT +
    process.photonIDrelisoToHLT +
    process.photonIDreliso2ToHLT
)    

##    ____       _   _     
##   |  _ \ __ _| |_| |__  
##   | |_) / _` | __| '_ \ 
##   |  __/ (_| | |_| | | |
##   |_|   \__,_|\__|_| |_|
##

if MC_flag:
    process.tagAndProbe = cms.Path(
        process.photon_sequence +
        process.ext_ToNearestJet_sequence + 
        process.allTagsAndProbes +
        process.mc_sequence + 
        process.tree_sequence
        )
else:
    process.tagAndProbe = cms.Path(
        process.photon_sequence +
        process.ext_ToNearestJet_sequence + 
        process.allTagsAndProbes +
        process.tree_sequence
        )

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(OUTPUT_FILE_NAME)
                                   )
