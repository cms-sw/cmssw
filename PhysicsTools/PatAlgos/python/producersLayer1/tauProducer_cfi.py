import FWCore.ParameterSet.Config as cms

patTaus = cms.EDProducer("PATTauProducer",
    # input
    tauSource = cms.InputTag("hpsPFTauProducer"),
    tauTransverseImpactParameterSource = cms.InputTag("hpsPFTauTransverseImpactParameters"),

    # add user data
    userData = cms.PSet(
      # add custom classes here
      userClasses = cms.PSet(
        src = cms.VInputTag('')
      ),
      # add doubles here
      userFloats = cms.PSet(
        src = cms.VInputTag('')
      ),
      # add ints here
      userInts = cms.PSet(
        src = cms.VInputTag('')
      ),
      # add candidate ptrs here
      userCands = cms.PSet(
        src = cms.VInputTag('')
      ),
      # add "inline" functions here
      userFunctions = cms.vstring(),
      userFunctionLabels = cms.vstring()
    ),

    # jet energy corrections
    addTauJetCorrFactors = cms.bool(False),
    tauJetCorrFactorsSource = cms.VInputTag(cms.InputTag("patTauJetCorrFactors")),

    # embedding objects (for Calo- and PFTaus)
    embedLeadTrack = cms.bool(False), ## embed in AOD externally stored leading track
    embedSignalTracks = cms.bool(False), ## embed in AOD externally stored signal tracks
    embedIsolationTracks = cms.bool(False), ## embed in AOD externally stored isolation tracks
    # embedding objects (for PFTaus only)
    embedLeadPFCand = cms.bool(False), ## embed in AOD externally stored leading PFCandidate
    embedLeadPFChargedHadrCand = cms.bool(False), ## embed in AOD externally stored leading PFChargedHadron candidate
    embedLeadPFNeutralCand = cms.bool(False), ## embed in AOD externally stored leading PFNeutral Candidate
    embedSignalPFCands = cms.bool(False), ## embed in AOD externally stored signal PFCandidates
    embedSignalPFChargedHadrCands = cms.bool(False), ## embed in AOD externally stored signal PFChargedHadronCandidates
    embedSignalPFNeutralHadrCands = cms.bool(False), ## embed in AOD externally stored signal PFNeutralHadronCandidates
    embedSignalPFGammaCands = cms.bool(False), ## embed in AOD externally stored signal PFGammaCandidates
    embedIsolationPFCands = cms.bool(False), ## embed in AOD externally stored isolation PFCandidates
    embedIsolationPFChargedHadrCands = cms.bool(False), ## embed in AOD externally stored isolation PFChargedHadronCandidates
    embedIsolationPFNeutralHadrCands = cms.bool(False), ## embed in AOD externally stored isolation PFNeutralHadronCandidates
    embedIsolationPFGammaCands = cms.bool(False), ## embed in AOD externally stored isolation PFGammaCandidates

    # embed IsoDeposits
    isoDeposits = cms.PSet(),

    # user defined isolation variables the variables defined here will be accessible
    # via pat::Tau::userIsolation(IsolationKeys key) with the key as defined in
    # DataFormats/PatCandidates/interface/Isolation.h
    userIsolation = cms.PSet(),

    # tau ID (for efficiency studies)
    addTauID     = cms.bool(True),
    tauIDSources = cms.PSet(
        # configure many IDs as InputTag <someName> = <someTag>
        # you can comment out those you don't want to save some
        # disk space
        decayModeFinding = cms.InputTag("hpsPFTauDiscriminationByDecayModeFinding"),
        decayModeFindingNewDMs =cms.InputTag("hpsPFTauDiscriminationByDecayModeFindingNewDMs"),
        chargedIsoPtSum = cms.InputTag("hpsPFTauChargedIsoPtSum"),
        neutralIsoPtSum = cms.InputTag("hpsPFTauNeutralIsoPtSum"),
        puCorrPtSum = cms.InputTag("hpsPFTauPUcorrPtSum"),
        footprintCorrection = cms.InputTag("hpsPFTauFootprintCorrection"),
        photonPtSumOutsideSignalCone = cms.InputTag("hpsPFTauPhotonPtSumOutsideSignalCone"),
        ##byIsolationMVA3oldDMwoLTraw = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA3oldDMwoLTraw'),
        ##byVLooseIsolationMVA3oldDMwoLT = cms.InputTag('hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT'),
        ##byLooseIsolationMVA3oldDMwoLT = cms.InputTag('hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwoLT'),
        ##byMediumIsolationMVA3oldDMwoLT = cms.InputTag('hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwoLT'),
        ##byTightIsolationMVA3oldDMwoLT = cms.InputTag('hpsPFTauDiscriminationByTightIsolationMVA3oldDMwoLT'),
        ##byVTightIsolationMVA3oldDMwoLT = cms.InputTag('hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwoLT'),
        ##byVVTightIsolationMVA3oldDMwoLT = cms.InputTag('hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwoLT'),                     
        byIsolationMVA3oldDMwLTraw = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA3oldDMwLTraw'),
        byVLooseIsolationMVA3oldDMwLT = cms.InputTag('hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT'),
        byLooseIsolationMVA3oldDMwLT = cms.InputTag('hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwLT'),
        byMediumIsolationMVA3oldDMwLT = cms.InputTag('hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwLT'),
        byTightIsolationMVA3oldDMwLT = cms.InputTag('hpsPFTauDiscriminationByTightIsolationMVA3oldDMwLT'),
        byVTightIsolationMVA3oldDMwLT = cms.InputTag('hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwLT'),
        byVVTightIsolationMVA3oldDMwLT = cms.InputTag('hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwLT'),                             
        ##byIsolationMVA3newDMwoLTraw = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA3newDMwoLTraw'),
        ##byVLooseIsolationMVA3newDMwoLT = cms.InputTag('hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT'),
        ##byLooseIsolationMVA3newDMwoLT = cms.InputTag('hpsPFTauDiscriminationByLooseIsolationMVA3newDMwoLT'),
        ##byMediumIsolationMVA3newDMwoLT = cms.InputTag('hpsPFTauDiscriminationByMediumIsolationMVA3newDMwoLT'),
        ##byTightIsolationMVA3newDMwoLT = cms.InputTag('hpsPFTauDiscriminationByTightIsolationMVA3newDMwoLT'),
        ##byVTightIsolationMVA3newDMwoLT = cms.InputTag('hpsPFTauDiscriminationByVTightIsolationMVA3newDMwoLT'),
        ##byVVTightIsolationMVA3newDMwoLT = cms.InputTag('hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwoLT'),                             
        byIsolationMVA3newDMwLTraw = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA3newDMwLTraw'),
        byVLooseIsolationMVA3newDMwLT = cms.InputTag('hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT'),
        byLooseIsolationMVA3newDMwLT = cms.InputTag('hpsPFTauDiscriminationByLooseIsolationMVA3newDMwLT'),
        byMediumIsolationMVA3newDMwLT = cms.InputTag('hpsPFTauDiscriminationByMediumIsolationMVA3newDMwLT'),
        byTightIsolationMVA3newDMwLT = cms.InputTag('hpsPFTauDiscriminationByTightIsolationMVA3newDMwLT'),
        byVTightIsolationMVA3newDMwLT = cms.InputTag('hpsPFTauDiscriminationByVTightIsolationMVA3newDMwLT'),
        byVVTightIsolationMVA3newDMwLT = cms.InputTag('hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwLT'),                             
        ##againstElectronLoose = cms.InputTag("hpsPFTauDiscriminationByLooseElectronRejection"),
        ##againstElectronMedium = cms.InputTag("hpsPFTauDiscriminationByMediumElectronRejection"),
        ##againstElectronTight = cms.InputTag("hpsPFTauDiscriminationByTightElectronRejection"),
        ##againstMuonLoose = cms.InputTag("hpsPFTauDiscriminationByLooseMuonRejection"),
        ##againstMuonMedium = cms.InputTag("hpsPFTauDiscriminationByMediumMuonRejection"),
        ##againstMuonTight = cms.InputTag("hpsPFTauDiscriminationByTightMuonRejection"),
        ##againstMuonLoose2 = cms.InputTag("hpsPFTauDiscriminationByLooseMuonRejection2"),
        ##againstMuonMedium2 = cms.InputTag("hpsPFTauDiscriminationByMediumMuonRejection2"),
        ##againstMuonTight2 = cms.InputTag("hpsPFTauDiscriminationByTightMuonRejection2"),
        againstMuonLoose3 = cms.InputTag("hpsPFTauDiscriminationByLooseMuonRejection3"),
        againstMuonTight3 = cms.InputTag("hpsPFTauDiscriminationByTightMuonRejection3"),
        ##againstMuonMVAraw = cms.InputTag('hpsPFTauDiscriminationByMVArawMuonRejection'),                                                            
        ##againstMuonLooseMVA = cms.InputTag('hpsPFTauDiscriminationByMVALooseMuonRejection'),
        ##againstMuonMediumMVA = cms.InputTag('hpsPFTauDiscriminationByMVAMediumMuonRejection'),
        ##againstMuonTightMVA = cms.InputTag('hpsPFTauDiscriminationByMVATightMuonRejection'),
        byLooseCombinedIsolationDeltaBetaCorr3Hits = cms.InputTag("hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits"),
        byMediumCombinedIsolationDeltaBetaCorr3Hits = cms.InputTag("hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits"),
        byTightCombinedIsolationDeltaBetaCorr3Hits = cms.InputTag("hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits"),
        byCombinedIsolationDeltaBetaCorrRaw3Hits = cms.InputTag("hpsPFTauDiscriminationByRawCombinedIsolationDBSumPtCorr3Hits"),
        byLoosePileupWeightedIsolation3Hits = cms.InputTag("hpsPFTauDiscriminationByLoosePileupWeightedIsolationDBSum3Hits"),
        byMediumPileupWeightedIsolation3Hits = cms.InputTag("hpsPFTauDiscriminationByMediumPileupWeightedIsolation3Hits"),
        byTightPileupWeightedIsolationHits = cms.InputTag("hpsPFTauDiscriminationByTightPileupWeightedIsolation3Hits"),
        byPhotonPtSumOutsideSignalCone = cms.InputTag("hpsPFTauDiscriminationByPhotonPtSumOutsideSignalCone"),
        byPileupWeightedIsolationRaw3Hits = cms.InputTag("hpsPFTauDiscriminationByRawPileupWeightedIsolation3Hits"),
        againstElectronMVA5raw = cms.InputTag("hpsPFTauDiscriminationByMVA5rawElectronRejection"),
        againstElectronMVA5category = cms.InputTag("hpsPFTauDiscriminationByMVA5rawElectronRejection:category"),
        againstElectronVLooseMVA5 = cms.InputTag("hpsPFTauDiscriminationByMVA5VLooseElectronRejection"),
        againstElectronLooseMVA5 = cms.InputTag("hpsPFTauDiscriminationByMVA5LooseElectronRejection"),
        againstElectronMediumMVA5 = cms.InputTag("hpsPFTauDiscriminationByMVA5MediumElectronRejection"),
        againstElectronTightMVA5 = cms.InputTag("hpsPFTauDiscriminationByMVA5TightElectronRejection"),
        againstElectronVTightMVA5 = cms.InputTag("hpsPFTauDiscriminationByMVA5VTightElectronRejection"),
        ##againstElectronDeadECAL = cms.InputTag("hpsPFTauDiscriminationByDeadECALElectronRejection"),        
      ),

    # mc matching configurables
    addGenMatch      = cms.bool(True),
    embedGenMatch    = cms.bool(True),
    genParticleMatch = cms.InputTag("tauMatch"),
    addGenJetMatch   = cms.bool(True),
    embedGenJetMatch = cms.bool(True),
    genJetMatch      = cms.InputTag("tauGenJetMatch"),

    # efficiencies
    addEfficiencies = cms.bool(False),
    efficiencies    = cms.PSet(),

    # resolution
    addResolutions  = cms.bool(False),
    resolutions     = cms.PSet()
)

