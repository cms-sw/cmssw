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
        neutralIsoPtSumWeight = cms.InputTag("hpsPFTauNeutralIsoPtSumWeight"),                  
        footprintCorrection = cms.InputTag("hpsPFTauFootprintCorrection"),
        photonPtSumOutsideSignalCone = cms.InputTag("hpsPFTauPhotonPtSumOutsideSignalCone"),
        byIsolationMVA3oldDMwLTraw = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA3oldDMwLTraw'),
        byVLooseIsolationMVA3oldDMwLT = cms.InputTag('hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT'),
        byLooseIsolationMVA3oldDMwLT = cms.InputTag('hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwLT'),
        byMediumIsolationMVA3oldDMwLT = cms.InputTag('hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwLT'),
        byTightIsolationMVA3oldDMwLT = cms.InputTag('hpsPFTauDiscriminationByTightIsolationMVA3oldDMwLT'),
        byVTightIsolationMVA3oldDMwLT = cms.InputTag('hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwLT'),
        byVVTightIsolationMVA3oldDMwLT = cms.InputTag('hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwLT'),                             
        byIsolationMVA3newDMwLTraw = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA3newDMwLTraw'),
        byVLooseIsolationMVA3newDMwLT = cms.InputTag('hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT'),
        byLooseIsolationMVA3newDMwLT = cms.InputTag('hpsPFTauDiscriminationByLooseIsolationMVA3newDMwLT'),
        byMediumIsolationMVA3newDMwLT = cms.InputTag('hpsPFTauDiscriminationByMediumIsolationMVA3newDMwLT'),
        byTightIsolationMVA3newDMwLT = cms.InputTag('hpsPFTauDiscriminationByTightIsolationMVA3newDMwLT'),
        byVTightIsolationMVA3newDMwLT = cms.InputTag('hpsPFTauDiscriminationByVTightIsolationMVA3newDMwLT'),
        byVVTightIsolationMVA3newDMwLT = cms.InputTag('hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwLT'),                             
        againstMuonLoose3 = cms.InputTag("hpsPFTauDiscriminationByLooseMuonRejection3"),
        againstMuonTight3 = cms.InputTag("hpsPFTauDiscriminationByTightMuonRejection3"),
        byLooseCombinedIsolationDeltaBetaCorr3Hits = cms.InputTag("hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits"),
        byMediumCombinedIsolationDeltaBetaCorr3Hits = cms.InputTag("hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits"),
        byTightCombinedIsolationDeltaBetaCorr3Hits = cms.InputTag("hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits"),
        byCombinedIsolationDeltaBetaCorrRaw3Hits = cms.InputTag("hpsPFTauDiscriminationByRawCombinedIsolationDBSumPtCorr3Hits"),
        byLoosePileupWeightedIsolation3Hits = cms.InputTag("hpsPFTauDiscriminationByLoosePileupWeightedIsolation3Hits"),
        byMediumPileupWeightedIsolation3Hits = cms.InputTag("hpsPFTauDiscriminationByMediumPileupWeightedIsolation3Hits"),
        byTightPileupWeightedIsolation3Hits = cms.InputTag("hpsPFTauDiscriminationByTightPileupWeightedIsolation3Hits"),
        byPhotonPtSumOutsideSignalCone = cms.InputTag("hpsPFTauDiscriminationByPhotonPtSumOutsideSignalCone"),
        byPileupWeightedIsolationRaw3Hits = cms.InputTag("hpsPFTauDiscriminationByRawPileupWeightedIsolation3Hits"),
        againstElectronMVA5raw = cms.InputTag("hpsPFTauDiscriminationByMVA5rawElectronRejection"),
        againstElectronMVA5category = cms.InputTag("hpsPFTauDiscriminationByMVA5rawElectronRejection:category"),
        againstElectronVLooseMVA5 = cms.InputTag("hpsPFTauDiscriminationByMVA5VLooseElectronRejection"),
        againstElectronLooseMVA5 = cms.InputTag("hpsPFTauDiscriminationByMVA5LooseElectronRejection"),
        againstElectronMediumMVA5 = cms.InputTag("hpsPFTauDiscriminationByMVA5MediumElectronRejection"),
        againstElectronTightMVA5 = cms.InputTag("hpsPFTauDiscriminationByMVA5TightElectronRejection"),
        againstElectronVTightMVA5 = cms.InputTag("hpsPFTauDiscriminationByMVA5VTightElectronRejection"),
        ##New Run2 MVA isolation
        byIsolationMVArun2v1DBoldDMwLTraw = cms.InputTag("DiscriminationByIsolationMVArun2v1DBoldDMwLTraw"),
        byVLooseIsolationMVArun2v1DBoldDMwLT = cms.InputTag("DiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT"),
        byLooseIsolationMVArun2v1DBoldDMwLT = cms.InputTag("DiscriminationByLooseIsolationMVArun2v1DBoldDMwLT"),
        byMediumIsolationMVArun2v1DBoldDMwLT = cms.InputTag("DiscriminationByMediumIsolationMVArun2v1DBoldDMwLT"),
        byTightIsolationMVArun2v1DBoldDMwLT = cms.InputTag("DiscriminationByTightIsolationMVArun2v1DBoldDMwLT"),
        byVTightIsolationMVArun2v1DBoldDMwLT = cms.InputTag("DiscriminationByVTightIsolationMVArun2v1DBoldDMwLT"),
        byVVTightIsolationMVArun2v1DBoldDMwLT = cms.InputTag("DiscriminationByVVTightIsolationMVArun2v1DBoldDMwLT"),
        byIsolationMVArun2v1DBnewDMwLTraw = cms.InputTag("DiscriminationByIsolationMVArun2v1DBnewDMwLTraw"),
        byVLooseIsolationMVArun2v1DBnewDMwLT = cms.InputTag("DiscriminationByVLooseIsolationMVArun2v1DBnewDMwLT"),
        byLooseIsolationMVArun2v1DBnewDMwLT = cms.InputTag("DiscriminationByLooseIsolationMVArun2v1DBnewDMwLT"),
        byMediumIsolationMVArun2v1DBnewDMwLT = cms.InputTag("DiscriminationByMediumIsolationMVArun2v1DBnewDMwLT"),
        byTightIsolationMVArun2v1DBnewDMwLT = cms.InputTag("DiscriminationByTightIsolationMVArun2v1DBnewDMwLT"),
        byVTightIsolationMVArun2v1DBnewDMwLT = cms.InputTag("DiscriminationByVTightIsolationMVArun2v1DBnewDMwLT"),
        byVVTightIsolationMVArun2v1DBnewDMwLT = cms.InputTag("DiscriminationByVVTightIsolationMVArun2v1DBnewDMwLT"),
        byIsolationMVArun2v1PWoldDMwLTraw = cms.InputTag("DiscriminationByIsolationMVArun2v1PWoldDMwLTraw"),
        byVLooseIsolationMVArun2v1PWoldDMwLT = cms.InputTag("DiscriminationByVLooseIsolationMVArun2v1PWoldDMwLT"),
        byLooseIsolationMVArun2v1PWoldDMwLT = cms.InputTag("DiscriminationByLooseIsolationMVArun2v1PWoldDMwLT"),
        byMediumIsolationMVArun2v1PWoldDMwLT = cms.InputTag("DiscriminationByMediumIsolationMVArun2v1PWoldDMwLT"),
        byTightIsolationMVArun2v1PWoldDMwLT = cms.InputTag("DiscriminationByTightIsolationMVArun2v1PWoldDMwLT"),
        byVTightIsolationMVArun2v1PWoldDMwLT = cms.InputTag("DiscriminationByVTightIsolationMVArun2v1PWoldDMwLT"),
        byVVTightIsolationMVArun2v1PWoldDMwLT = cms.InputTag("DiscriminationByVVTightIsolationMVArun2v1PWoldDMwLT"),
        byIsolationMVArun2v1PWnewDMwLTraw = cms.InputTag("DiscriminationByIsolationMVArun2v1PWnewDMwLTraw"),
        byVLooseIsolationMVArun2v1PWnewDMwLT = cms.InputTag("DiscriminationByVLooseIsolationMVArun2v1PWnewDMwLT"),
        byLooseIsolationMVArun2v1PWnewDMwLT = cms.InputTag("DiscriminationByLooseIsolationMVArun2v1PWnewDMwLT"),
        byMediumIsolationMVArun2v1PWnewDMwLT = cms.InputTag("DiscriminationByMediumIsolationMVArun2v1PWnewDMwLT"),
        byTightIsolationMVArun2v1PWnewDMwLT = cms.InputTag("DiscriminationByTightIsolationMVArun2v1PWnewDMwLT"),
        byVTightIsolationMVArun2v1PWnewDMwLT = cms.InputTag("DiscriminationByVTightIsolationMVArun2v1PWnewDMwLT"),
        byVVTightIsolationMVArun2v1PWnewDMwLT = cms.InputTag("DiscriminationByVVTightIsolationMVArun2v1PWnewDMwLT"),
        chargedIsoPtSumdR03 = cms.InputTag("ChargedIsoPtSumdR03"),
        neutralIsoPtSumdR03 = cms.InputTag("NeutralIsoPtSumdR03"),
        puCorrPtSumdR03 = cms.InputTag("PUcorrPtSumdR03"),
        neutralIsoPtSumWeightdR03 = cms.InputTag("NeutralIsoPtSumWeightdR03"),
        footprintCorrectiondR03 = cms.InputTag("FootprintCorrectiondR03"),
        photonPtSumOutsideSignalConedR03 = cms.InputTag("PhotonPtSumOutsideSignalConedR03"),
        byIsolationMVArun2v1DBdR03oldDMwLTraw = cms.InputTag("DiscriminationByIsolationMVArun2v1DBdR03oldDMwLTraw"),
        byVLooseIsolationMVArun2v1DBdR03oldDMwLT = cms.InputTag("DiscriminationByVLooseIsolationMVArun2v1DBdR03oldDMwLT"),
        byLooseIsolationMVArun2v1DBdR03oldDMwLT = cms.InputTag("DiscriminationByLooseIsolationMVArun2v1DBdR03oldDMwLT"),
        byMediumIsolationMVArun2v1DBdR03oldDMwLT = cms.InputTag("DiscriminationByMediumIsolationMVArun2v1DBdR03oldDMwLT"),
        byTightIsolationMVArun2v1DBdR03oldDMwLT = cms.InputTag("DiscriminationByTightIsolationMVArun2v1DBdR03oldDMwLT"),
        byVTightIsolationMVArun2v1DBdR03oldDMwLT = cms.InputTag("DiscriminationByVTightIsolationMVArun2v1DBdR03oldDMwLT"),
        byVVTightIsolationMVArun2v1DBdR03oldDMwLT = cms.InputTag("DiscriminationByVVTightIsolationMVArun2v1DBdR03oldDMwLT"),
        byIsolationMVArun2v1PWdR03oldDMwLTraw = cms.InputTag("DiscriminationByIsolationMVArun2v1PWdR03oldDMwLTraw"),
        byVLooseIsolationMVArun2v1PWdR03oldDMwLT = cms.InputTag("DiscriminationByVLooseIsolationMVArun2v1PWdR03oldDMwLT"),
        byLooseIsolationMVArun2v1PWdR03oldDMwLT = cms.InputTag("DiscriminationByLooseIsolationMVArun2v1PWdR03oldDMwLT"),
        byMediumIsolationMVArun2v1PWdR03oldDMwLT = cms.InputTag("DiscriminationByMediumIsolationMVArun2v1PWdR03oldDMwLT"),
        byTightIsolationMVArun2v1PWdR03oldDMwLT = cms.InputTag("DiscriminationByTightIsolationMVArun2v1PWdR03oldDMwLT"),
        byVTightIsolationMVArun2v1PWdR03oldDMwLT = cms.InputTag("DiscriminationByVTightIsolationMVArun2v1PWdR03oldDMwLT"),
        byVVTightIsolationMVArun2v1PWdR03oldDMwLT = cms.InputTag("DiscriminationByVVTightIsolationMVArun2v1PWdR03oldDMwLT"),
        ##New Run2 MVA discriminator against electrons
        againstElectronMVA6raw = cms.InputTag("DiscriminationByMVA6rawElectronRejection"),
        againstElectronMVA6category = cms.InputTag("DiscriminationByMVA6rawElectronRejection:category"),
        againstElectronVLooseMVA6 = cms.InputTag("DiscriminationByMVA6VLooseElectronRejection"),
        againstElectronLooseMVA6 = cms.InputTag("DiscriminationByMVA6LooseElectronRejection"),
        againstElectronMediumMVA6 = cms.InputTag("DiscriminationByMVA6MediumElectronRejection"),
        againstElectronTightMVA6 = cms.InputTag("DiscriminationByMVA6TightElectronRejection"),
        againstElectronVTightMVA6 = cms.InputTag("DiscriminationByMVA6VTightElectronRejection"),
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

