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
HLTPath1 = "HLT_Photon70_Cleaned_L1R_v1"
HLTPath2 = "HLT_Photon50_Cleaned_L1R_v1"
HLTPath3 = "HLT_Photon30_Cleaned_L1R"
#InputTagProcess = "REDIGI36X"
InputTagProcess = "HLT"
RECOProcess = "RECO"
JET_COLL = "ak5PFJets"
JET_CUTS = "abs(eta)<2.6 && chargedHadronEnergyFraction>0 && electronEnergyFraction<0.1 && nConstituents>1 && neutralHadronEnergyFraction<0.99 && neutralEmEnergyFraction<0.99 && pt>15.0" 

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
    srcObjectSelection = cms.string("hadronicOverEm<0.5 && pt>10 && abs(eta)<1.479"),
    srcObjectsToMatchSelection = cms.string('pt > 20.0 && quality("highPurity")'),                                            
    deltaRMax = cms.double(0.1)
    )

##     ___           _       _   _             
##    |_ _|___  ___ | | __ _| |_(_) ___  _ __  
##     | |/ __|/ _ \| |/ _` | __| |/ _ \| '_ \ 
##     | |\__ \ (_) | | (_| | |_| | (_) | | | |
##    |___|___/\___/|_|\__,_|\__|_|\___/|_| |_|

                                         
#  Isolation ################
#ECAL and HCAL only
process.probePhotonsPassingIsolation = cms.EDFilter("PhotonRefSelector",
    src = cms.InputTag("probePhotons"),
    cut = cms.string("(ecalRecHitSumEtConeDR04 < (0.006*pt + 4.2)) && (hcalTowerSumEtConeDR04 < (0.0025*pt + 2.2 ))")
    )

##  ____  _           _                ___    _ 
## |  _ \| |__   ___ | |_ ___  _ __   |_ _|__| |
## | |_) | '_ \ / _ \| __/ _ \| '_ \   | |/ _` |
## |  __/| | | | (_) | || (_) | | | |  | | (_| |
## |_|   |_| |_|\___/ \__\___/|_| |_| |___\__,_|
##        
#track isolation
process.probePhotonsPassingId = process.probePhotonsPassingIsolation.clone()
process.probePhotonsPassingId.cut = cms.string(
    process.probePhotonsPassingIsolation.cut.value() +
    " && (hadronicOverEm < 0.05) && (trkSumPtHollowConeDR04 < (0.001*pt + 2.0)"
    " && (sigmaIetaIeta < 0.013))")
                         
##    _____     _                         __  __       _       _     _             
##   |_   _| __(_) __ _  __ _  ___ _ __  |  \/  | __ _| |_ ___| |__ (_)_ __   __ _ 
##     | || '__| |/ _` |/ _` |/ _ \ '__| | |\/| |/ _` | __/ __| '_ \| | '_ \ / _` |
##     | || |  | | (_| | (_| |  __/ |    | |  | | (_| | || (__| | | | | | | | (_| |
##     |_||_|  |_|\__, |\__, |\___|_|    |_|  |_|\__,_|\__\___|_| |_|_|_| |_|\__, |
##                |___/ |___/                                                |___/ 
##   
process.probePhotonsPassingHLT = cms.EDProducer(
    "trgMatchedPhotonProducer",                     
    InputProducer = cms.InputTag("probePhotonsPassingId"),
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

#step 1: tag should be tightly matched to a track
process.trackMatchedPhotons = process.probePhotons.clone()


#step 2: tag should have good shower shape, a pixel seed, have good H/E, be reasonably high pT, and be in 
process.goodPhotons = cms.EDFilter(
    "PhotonRefSelector",
    src = cms.InputTag("trackMatchedPhotons"),
    cut = cms.string("(sigmaIetaIeta < 0.009) && (hasPixelSeed = 1.0) && (hadronicOverEm < 0.05) && (pt > 30.0)"
                     " && (abs(eta)<1.479) && (abs(abs(superCluster.eta) - 1.479)>=0.1)"
                     )
    )

#step 3: tag should have fired the HLT path under study
process.Tag = cms.EDProducer("trgMatchedPhotonProducer",                     
    InputProducer = cms.InputTag("goodPhotons"),
    hltTags = cms.VInputTag(
    cms.InputTag(HLTPath1,"",InputTagProcess),
    cms.InputTag(HLTPath2,"",InputTagProcess),
    cms.InputTag(HLTPath3,"",InputTagProcess)
    ),
    triggerEventTag = cms.untracked.InputTag("hltTriggerSummaryAOD","",InputTagProcess),
    triggerResultsTag = cms.untracked.InputTag("TriggerResults", "", InputTagProcess)
    )


process.photon_sequence = cms.Sequence(
    process.probePhotons +
    process.probePhotonsPassingIsolation +
    process.probePhotonsPassingId +
    process.probePhotonsPassingHLT + 
    process.trackMatchedPhotons + process.goodPhotons + process.Tag
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
process.tagIsoPhotons = process.tagPhoton.clone()
process.tagIsoPhotons.decay = cms.string("Tag probePhotonsPassingIsolation")
process.tagIdPhotons = process.tagPhoton.clone()
process.tagIdPhotons.decay = cms.string("Tag probePhotonsPassingId")
process.tagHLTPhotons = process.tagPhoton.clone()
process.tagHLTPhotons.decay = cms.string("Tag probePhotonsPassingHLT")

process.allTagsAndProbes = cms.Sequence(
    process.tagPhoton + process.tagIsoPhotons + process.tagIdPhotons + process.tagHLTPhotons
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
process.McMatchIso = process.McMatchTag.clone()
process.McMatchIso.src = cms.InputTag("probePhotonsPassingIsolation")
process.McMatchId = process.McMatchTag.clone()
process.McMatchId.src = cms.InputTag("probePhotonsPassingId")
process.McMatchHLT = process.McMatchTag.clone()
process.McMatchHLT.src = cms.InputTag("probePhotonsPassingHLT")


process.mc_sequence = cms.Sequence(
   process.McMatchTag +  process.McMatchPhoton +
   process.McMatchIso +
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


TagPhotonVariablesToStore = cms.PSet(
    photon_eta = cms.string("eta"),
    photon_pt  = cms.string("pt"),
    photon_phi  = cms.string("phi"),
    photon_px  = cms.string("px"),
    photon_py  = cms.string("py"),
    photon_pz  = cms.string("pz"),
    ## super cluster quantities
    sc_energy = cms.string("superCluster.energy"),
    sc_et     = cms.string("superCluster.energy*sin(superCluster.position.theta)"),    
    sc_x      = cms.string("superCluster.x"),
    sc_y      = cms.string("superCluster.y"),
    sc_z      = cms.string("superCluster.z"),
    sc_eta    = cms.string("superCluster.eta"),
    sc_phi    = cms.string("superCluster.phi"),
    sc_size   = cms.string("superCluster.size"), # number of hits
    sc_rawEnergy = cms.string("superCluster.rawEnergy"), 
    sc_preshowerEnergy   = cms.string("superCluster.preshowerEnergy"), 
    sc_phiWidth   = cms.string("superCluster.phiWidth"), 
    sc_etaWidth   = cms.string("superCluster.etaWidth"),         
    ## isolation 
    photon_trackiso_dr04 = cms.string("trkSumPtHollowConeDR04"),
    photon_ecaliso_dr04  = cms.string("ecalRecHitSumEtConeDR04"),
    photon_hcaliso_dr04  = cms.string("hcalTowerSumEtConeDR04"),
    photon_trackiso_dr03 = cms.string("trkSumPtHollowConeDR03"),
    photon_ecaliso_dr03  = cms.string("ecalRecHitSumEtConeDR03"),
    photon_hcaliso_dr03  = cms.string("hcalTowerSumEtConeDR04"),
    ## classification, location, etc.    
    photon_isEB           = cms.string("isEB"),
    photon_isEE           = cms.string("isEE"),
    photon_isEBEEGap      = cms.string("isEBEEGap"),
    photon_isEBEtaGap     = cms.string("isEBEtaGap"),
    photon_isEBPhiGap     = cms.string("isEBPhiGap"),
    photon_isEEDeeGap     = cms.string("isEEDeeGap"),
    photon_isEERingGap    = cms.string("isEERingGap"),
    ## Hcal energy over Ecal Energy
    photon_HoverE         = cms.string("hadronicOverEm"),
    photon_HoverE_Depth1  = cms.string("hadronicDepth1OverEm"),
    photon_HoverE_Depth2  = cms.string("hadronicDepth2OverEm"),
    ## Cluster shape information
    photon_sigmaEtaEta  = cms.string("sigmaEtaEta"),
    photon_sigmaIetaIeta = cms.string("sigmaIetaIeta"),
    photon_e1x5               = cms.string("e1x5"),
    photon_e2x5            = cms.string("e2x5"),
    photon_e5x5               = cms.string("e5x5"),
    photon_hasPixelSeed = cms.string("hasPixelSeed")
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



##     ____      __       __    ___           
##    / ___|___ / _|      \ \  |_ _|___  ___  
##   | |  _/ __| |_   _____\ \  | |/ __|/ _ \ 
##   | |_| \__ \  _| |_____/ /  | |\__ \ (_) |
##    \____|___/_|        /_/  |___|___/\___/ 
##   

## loose photon --> isolation
process.PhotonToIsolation = cms.EDAnalyzer("TagProbeFitTreeProducer",
    ## pick the defaults
    mcTruthCommonStuff,
    CommonStuffForPhotonProbe,
    # choice of tag and probe pairs, and arbitration                 
    tagProbePairs = cms.InputTag("tagPhoton"),
    arbitration   = cms.string("None"),                      
    flags = cms.PSet(
        probe_passingIso = cms.InputTag("probePhotonsPassingIsolation"),
        probe_passingHLT = cms.InputTag("probePhotonsPassingHLT"),
        probe_passingId = cms.InputTag("probePhotonsPassingId"),
    ),
    probeMatches  = cms.InputTag("McMatchPhoton"),
    allProbes     = cms.InputTag("probePhotons")
)
process.PhotonToIsolation.variables.probe_dRjet = cms.InputTag("photonDRToNearestJet")
process.PhotonToIsolation.variables.probe_nJets = cms.InputTag("JetMultiplicity")

##    ___                 __    ___    _ 
##   |_ _|___  ___        \ \  |_ _|__| |
##    | |/ __|/ _ \   _____\ \  | |/ _` |
##    | |\__ \ (_) | |_____/ /  | | (_| |
##   |___|___/\___/       /_/  |___\__,_|
##   
#isolated --> ID'ed photon
process.IsoToId = process.PhotonToIsolation.clone()
process.IsoToId.tagProbePairs = cms.InputTag("tagIsoPhotons")
process.IsoToId.probeMatches  = cms.InputTag("McMatchIso")
process.IsoToId.allProbes     = cms.InputTag("probePhotonsPassingIsolation")



##    ___    _       __    _   _ _   _____ 
##   |_ _|__| |      \ \  | | | | | |_   _|
##    | |/ _` |  _____\ \ | |_| | |   | |  
##    | | (_| | |_____/ / |  _  | |___| |  
##   |___\__,_|      /_/  |_| |_|_____|_|  

#ID'ed --> HLT photon
process.IdToHLT = process.PhotonToIsolation.clone()
process.IdToHLT.tagProbePairs = cms.InputTag("tagIdPhotons")
process.IdToHLT.probeMatches  = cms.InputTag("McMatchId")
process.IdToHLT.allProbes     = cms.InputTag("probePhotonsPassingId")



#loose --> HLT photon
process.PhotonToHLT = process.PhotonToIsolation.clone()
process.PhotonToHLT.tagProbePairs = cms.InputTag("tagPhoton")
process.PhotonToHLT.probeMatches  = cms.InputTag("McMatchPhoton")
process.PhotonToHLT.allProbes     = cms.InputTag("probePhotons")


process.tree_sequence = cms.Sequence(
    process.PhotonToIsolation +
    process.IsoToId + process.IdToHLT + process.PhotonToHLT
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
