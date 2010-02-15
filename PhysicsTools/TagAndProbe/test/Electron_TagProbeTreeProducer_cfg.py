import FWCore.ParameterSet.Config as cms

process = cms.Process("TagProbe")

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
#process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.source = cms.Source("PoolSource", 
    fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_3_4_1/RelValZEE/GEN-SIM-RECO/MC_3XY_V14-v1/0004/EA38A37E-B5ED-DE11-8DD2-000423D6CA6E.root',
       '/store/relval/CMSSW_3_4_1/RelValZEE/GEN-SIM-RECO/MC_3XY_V14-v1/0004/A2951E1F-8FED-DE11-B23A-001D09F29114.root',
       '/store/relval/CMSSW_3_4_1/RelValZEE/GEN-SIM-RECO/MC_3XY_V14-v1/0004/9EC0D2FB-90ED-DE11-80C5-000423D8FA38.root',
       '/store/relval/CMSSW_3_4_1/RelValZEE/GEN-SIM-RECO/MC_3XY_V14-v1/0004/6C21EC4A-90ED-DE11-9F3A-0019B9F707D8.root',
       '/store/relval/CMSSW_3_4_1/RelValZEE/GEN-SIM-RECO/MC_3XY_V14-v1/0004/4E4D206B-8DED-DE11-966A-001617E30F50.root'
    )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )    



#  SuperClusters  ################
process.superClusters = cms.EDFilter("SuperClusterMerger",
   src = cms.VInputTag(cms.InputTag("hybridSuperClusters","", "RECO"),
                       cms.InputTag("multi5x5SuperClustersWithPreshower","", "RECO"))  
)

process.superClusterCands = cms.EDProducer("ConcreteEcalCandidateProducer",
   src = cms.InputTag("superClusters"),
   particleType = cms.int32(11),
   cut = cms.string("((energy)*sin(position.theta)>20.0) && (abs(eta)<2.5) && !(1.4442<abs(eta)<1.560)")
)

## process.superClusterCands = cms.EDFilter("CandViewSelector",
##    src = cms.InputTag("SCtoCandidate"),
##    cut = cms.string('et > 0.0'),
##    filter = cms.bool(True)
## )

## process.Sc2GsfMatching = cms.EDFilter("TrivialDeltaRViewMatcher",
##     src = cms.InputTag("superClusterCands"),
##     distMin = cms.double(1.0),
##     matched = cms.InputTag("gsfElectrons")
## )

## # Use the producer to get a list of matched candidates
## process.SCPassingGsf = cms.EDFilter("MatchedCandidateSelector",
##      match = cms.InputTag("Sc2GsfMatching"),
##      src = cms.InputTag("superClusterCands")
## )




## process.SCPassingGsf = cms.EDProducer("MatchedCandidateSelector",
##     src   = cms.InputTag("gsfElectrons"),
##     match = cms.InputTag("Sc2GsfMatching"),
## )



process.sc_sequence = cms.Sequence( process.superClusters *
                                    process.superClusterCands
                                    )


#  GsfElectron ################ 
process.PassingGsf = cms.EDFilter("GsfElectronRefSelector",
    src = cms.InputTag("gsfElectrons"),
    cut = cms.string("(isEB || isEE) && (ecalEnergy*sin(superClusterPosition.theta)>20.0)")    
)



#  Isolation ################ 
process.PassingIsolation = cms.EDFilter("GsfElectronRefSelector",
    src = cms.InputTag("gsfElectrons"),
    cut = cms.string(process.PassingGsf.cut.value() +
         " && (dr04TkSumPt/pt<0.2) && (dr04EcalRecHitSumEt/et<0.2) && (dr04HcalTowerSumEt/et<0.2)")  
)

##    ____               _               ____            _                  ___ ____  
##   |  _ \ __ _ ___ ___(_)_ __   __ _  |  _ \ _ __ ___ | |__   ___  ___ _  |_ _|  _ \ 
##   | |_) / _` / __/ __| | '_ \ / _` | | |_) | '__/ _ \| '_ \ / _ \/ __(_)  | || | | |
##   |  __/ (_| \__ \__ \ | | | | (_| | |  __/| | | (_) | |_) |  __/\__ \_   | || |_| |
##   |_|   \__,_|___/___/_|_| |_|\__, | |_|   |_|  \___/|_.__/ \___||___(_) |___|____/ 
##                               |___/                                                                
##   
# Electron ID  ######
process.PassingId = cms.EDFilter("GsfElectronRefSelector",
    src = cms.InputTag("gsfElectrons"),
    cut = cms.string(process.PassingIsolation.cut.value() +
          " && ( (isEB && sigmaIetaIeta<0.01 && deltaEtaSuperClusterTrackAtVtx<0.0071)"
          "|| (isEE && sigmaIetaIeta<0.028 && deltaEtaSuperClusterTrackAtVtx<0.0066) )")   
)

##    ____               _               ____            _                   _____     _                       
##   |  _ \ __ _ ___ ___(_)_ __   __ _  |  _ \ _ __ ___ | |__   ___  ___ _  |_   _| __(_) __ _  __ _  ___ _ __ 
##   | |_) / _` / __/ __| | '_ \ / _` | | |_) | '__/ _ \| '_ \ / _ \/ __(_)   | || '__| |/ _` |/ _` |/ _ \ '__|
##   |  __/ (_| \__ \__ \ | | | | (_| | |  __/| | | (_) | |_) |  __/\__ \_    | || |  | | (_| | (_| |  __/ |   
##   |_|   \__,_|___/___/_|_| |_|\__, | |_|   |_|  \___/|_.__/ \___||___(_)   |_||_|  |_|\__, |\__, |\___|_|   
##                               |___/                                                   |___/ |___/           
##  
# Trigger  ##################

process.PassingHLT = cms.EDProducer("eTriggerGsfElectronCollection",                     
    InputProducer = cms.InputTag("PassingId"),                          
    hltTag = cms.untracked.InputTag("HLT_Ele15_SW_LooseTrackIso_L1R","","HLT"),
    triggerEventTag = cms.untracked.InputTag("hltTriggerSummaryAOD","","HLT")
)




process.Tag = process.PassingId.clone()
process.ele_sequence = cms.Sequence(
    process.PassingGsf * 
    process.PassingIsolation * process.PassingId * 
    process.PassingHLT * process.Tag
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
    decay = cms.string("Tag@+ superClusterCands@-"), # charge coniugate states are implied
    cut   = cms.string("60 < mass < 120"),
)


process.tagGsf = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("Tag@+ PassingGsf@-"), # charge coniugate states are implied
    cut   = cms.string("60 < mass < 120"),
)


process.tagIso = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("Tag@+ PassingIsolation@-"), # charge coniugate states are implied
    cut   = cms.string("60 < mass < 120"),
)


process.tagId = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("Tag@+ PassingId@-"), # charge coniugate states are implied
    cut   = cms.string("60 < mass < 120"),
)


process.tagHLT = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("Tag@+ PassingHLT@-"), # charge coniugate states are implied
    cut   = cms.string("60 < mass < 120"),
)



process.allTagsAndProbes = cms.Sequence(
    process.tagSC * process.tagGsf *
    process.tagIso * process.tagId * process.tagHLT
)


##    __  __  ____   __  __       _       _               
##   |  \/  |/ ___| |  \/  | __ _| |_ ___| |__   ___  ___ 
##   | |\/| | |     | |\/| |/ _` | __/ __| '_ \ / _ \/ __|
##   | |  | | |___  | |  | | (_| | || (__| | | |  __/\__ \
##   |_|  |_|\____| |_|  |_|\__,_|\__\___|_| |_|\___||___/
##                                                        

process.McMatchTag = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(11),
    src = cms.InputTag("Tag"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("genParticles")
)


process.McMatchSC = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(11),
    src = cms.InputTag("superClusterCands"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("genParticles")
)

process.McMatchGsf = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(11),
    src = cms.InputTag("PassingGsf"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("genParticles")
)

process.McMatchIso = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(11),
    src = cms.InputTag("PassingIsolation"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("genParticles")
)

process.McMatchId = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(11),
    src = cms.InputTag("PassingId"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("genParticles")
)

process.McMatchHLT = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(11),
    src = cms.InputTag("PassingHLT"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("genParticles")
)



process.mc_sequence = cms.Sequence(
   process.McMatchTag + process.McMatchSC +
   process.McMatchGsf + process.McMatchIso +
   process.McMatchId  + process.McMatchHLT
)


##    _____           _       _ ____            _            _   _  ____ 
##   |_   _|_ _  __ _( )_ __ ( )  _ \ _ __ ___ | |__   ___  | \ | |/ ___|
##     | |/ _` |/ _` |/| '_ \|/| |_) | '__/ _ \| '_ \ / _ \ |  \| | |  _ 
##     | | (_| | (_| | | | | | |  __/| | | (_) | |_) |  __/ | |\  | |_| |
##     |_|\__,_|\__, | |_| |_| |_|   |_|  \___/|_.__/ \___| |_| \_|\____|
##              |___/                                                    
##

## super cluster --> gsf electron
process.SCToGsf = cms.EDAnalyzer("TagProbeFitTreeProducer",
    tagProbePairs = cms.InputTag("tagSC"),
    arbitration   = cms.string("OneProbe"),
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
    flags = cms.PSet(
        passing = cms.InputTag("PassingGsf")
        #passing = cms.InputTag("SCPassingGsf")
    ),
    isMC = cms.bool(True),
    tagMatches = cms.InputTag("McMatchTag"),
    probeMatches  = cms.InputTag("McMatchSC"),
    motherPdgId = cms.vint32(22,23),
    makeMCUnbiasTree = cms.bool(True),
    checkMotherInUnbiasEff = cms.bool(True),
    allProbes     = cms.InputTag("superClusterCands")
)

##  gsf electron --> isolation
process.GsfToIso = cms.EDAnalyzer("TagProbeFitTreeProducer",
    tagProbePairs = cms.InputTag("tagGsf"),
    arbitration   = cms.string("OneProbe"),
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
    flags = cms.PSet(
        passing = cms.InputTag("PassingIsolation")
    ),
    isMC = cms.bool(True),
    tagMatches = cms.InputTag("McMatchTag"),
    probeMatches  = cms.InputTag("McMatchGsf"),
    motherPdgId = cms.vint32(22,23),
    makeMCUnbiasTree = cms.bool(True),
    checkMotherInUnbiasEff = cms.bool(True),
    allProbes     = cms.InputTag("PassingGsf")
)

##  isolation --> Id
process.IsoToId = cms.EDAnalyzer("TagProbeFitTreeProducer",
    tagProbePairs = cms.InputTag("tagIso"),
    arbitration   = cms.string("OneProbe"),
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
    flags = cms.PSet(
        passing = cms.InputTag("PassingId")
    ),
    isMC = cms.bool(True),
    tagMatches = cms.InputTag("McMatchTag"),
    probeMatches  = cms.InputTag("McMatchIso"),
    motherPdgId = cms.vint32(22,23),
    makeMCUnbiasTree = cms.bool(True),
    checkMotherInUnbiasEff = cms.bool(True),
    allProbes     = cms.InputTag("PassingIsolation")
)

##  Id --> HLT
process.IdToHLT = cms.EDAnalyzer("TagProbeFitTreeProducer",
    tagProbePairs = cms.InputTag("tagId"),
    arbitration   = cms.string("OneProbe"),
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
    flags = cms.PSet(
        passing = cms.InputTag("PassingHLT")
    ),
    isMC = cms.bool(True),
    tagMatches = cms.InputTag("McMatchTag"),
    probeMatches  = cms.InputTag("McMatchId"),
    motherPdgId = cms.vint32(22,23),
    makeMCUnbiasTree = cms.bool(True),
    checkMotherInUnbiasEff = cms.bool(True),
    allProbes     = cms.InputTag("PassingId")
)


process.tree_sequence = cms.Sequence(
    process.SCToGsf + process.GsfToIso +
    process.IsoToId + process.IdToHLT
)    



## process.tree_sequence = cms.Sequence(
##     process.GsfToIso
## )    



##    ____       _   _     
##   |  _ \ __ _| |_| |__  
##   | |_) / _` | __| '_ \ 
##   |  __/ (_| | |_| | | |
##   |_|   \__,_|\__|_| |_|
##

process.tagAndProbe = cms.Path(
    process.sc_sequence * process.ele_sequence * 
    process.allTagsAndProbes * process.mc_sequence * 
    process.tree_sequence
)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("testNewWrite.root")
                                   )
