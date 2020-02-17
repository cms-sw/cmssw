import FWCore.ParameterSet.Config as cms

import RecoTauTag.Configuration.HPSPFTaus_cff as RecoModules #Working point indices are extracted from here

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
        decayModeFinding = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByDecayModeFinding"), provenanceConfigLabel=cms.string(""), idLabel=cms.string("")),
        decayModeFindingNewDMs = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByDecayModeFindingNewDMs"), provenanceConfigLabel=cms.string(""), idLabel=cms.string("")),
        chargedIsoPtSum = cms.PSet( inputTag = cms.InputTag("hpsPFTauBasicDiscriminators"), provenanceConfigLabel=cms.string("IDdefinitions"), idLabel=cms.string("ChargedIsoPtSum")),
        neutralIsoPtSum = cms.PSet( inputTag = cms.InputTag("hpsPFTauBasicDiscriminators"), provenanceConfigLabel=cms.string("IDdefinitions"), idLabel=cms.string("NeutralIsoPtSum")),
        puCorrPtSum = cms.PSet( inputTag = cms.InputTag("hpsPFTauBasicDiscriminators"), provenanceConfigLabel=cms.string("IDdefinitions"), idLabel=cms.string("PUcorrPtSum")),
        neutralIsoPtSumWeight = cms.PSet( inputTag = cms.InputTag("hpsPFTauBasicDiscriminators"), provenanceConfigLabel=cms.string("IDdefinitions"), idLabel=cms.string("NeutralIsoPtSumWeight")),                  
        footprintCorrection = cms.PSet( inputTag = cms.InputTag("hpsPFTauBasicDiscriminators"), provenanceConfigLabel=cms.string("IDdefinitions"), idLabel=cms.string("TauFootprintCorrection")),
        photonPtSumOutsideSignalCone = cms.PSet( inputTag = cms.InputTag("hpsPFTauBasicDiscriminators"), provenanceConfigLabel=cms.string("IDdefinitions"), idLabel=cms.string("PhotonPtSumOutsideSignalCone")),
        againstMuonLoose3 = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByMuonRejection3"), provenanceConfigLabel=cms.string("IDWPdefinitions"), idLabel=cms.string("ByLooseMuonRejection3")),
        againstMuonTight3 = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByMuonRejection3"), provenanceConfigLabel=cms.string("IDWPdefinitions"), idLabel=cms.string("ByTightMuonRejection3")),
        byLooseCombinedIsolationDeltaBetaCorr3Hits = cms.PSet( inputTag = cms.InputTag("hpsPFTauBasicDiscriminators"), provenanceConfigLabel=cms.string("IDWPdefinitions"), idLabel=cms.string("ByLooseCombinedIsolationDBSumPtCorr3Hits")),
        byMediumCombinedIsolationDeltaBetaCorr3Hits = cms.PSet( inputTag = cms.InputTag("hpsPFTauBasicDiscriminators"), provenanceConfigLabel=cms.string("IDWPdefinitions"), idLabel=cms.string("ByMediumCombinedIsolationDBSumPtCorr3Hits")),
        byTightCombinedIsolationDeltaBetaCorr3Hits = cms.PSet( inputTag = cms.InputTag("hpsPFTauBasicDiscriminators"), provenanceConfigLabel=cms.string("IDWPdefinitions"), idLabel=cms.string("ByTightCombinedIsolationDBSumPtCorr3Hits")),
        byCombinedIsolationDeltaBetaCorrRaw3Hits = cms.PSet( inputTag = cms.InputTag("hpsPFTauBasicDiscriminators"), provenanceConfigLabel=cms.string("IDdefinitions"), idLabel=cms.string("ByRawCombinedIsolationDBSumPtCorr3Hits")),
        byPhotonPtSumOutsideSignalCone = cms.PSet( inputTag = cms.InputTag("hpsPFTauBasicDiscriminators"), provenanceConfigLabel=cms.string("IDWPdefinitions"), idLabel=cms.string("ByPhotonPtSumOutsideSignalCone")),
        byIsolationMVArun2v1DBoldDMwLTraw = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLT"), provenanceConfigLabel=cms.string("rawValues"), idLabel=cms.string("discriminator")),
        byVVLooseIsolationMVArun2v1DBoldDMwLT = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLT"), provenanceConfigLabel=cms.string("workingPoints"), idLabel=cms.string("_VVLoose")),
        byVLooseIsolationMVArun2v1DBoldDMwLT = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLT"), provenanceConfigLabel=cms.string("workingPoints"), idLabel=cms.string("_VLoose")),
        byLooseIsolationMVArun2v1DBoldDMwLT = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLT"), provenanceConfigLabel=cms.string("workingPoints"), idLabel=cms.string("_Loose")),
        byMediumIsolationMVArun2v1DBoldDMwLT = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLT"), provenanceConfigLabel=cms.string("workingPoints"), idLabel=cms.string("_Medium")),
        byTightIsolationMVArun2v1DBoldDMwLT = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLT"), provenanceConfigLabel=cms.string("workingPoints"), idLabel=cms.string("_Tight")),
        byVTightIsolationMVArun2v1DBoldDMwLT = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLT"), provenanceConfigLabel=cms.string("workingPoints"), idLabel=cms.string("_VTight")),
        byVVTightIsolationMVArun2v1DBoldDMwLT = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLT"), provenanceConfigLabel=cms.string("workingPoints"), idLabel=cms.string("_VVTight")),
        byIsolationMVArun2v1DBnewDMwLTraw = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLT"), provenanceConfigLabel=cms.string("rawValues"), idLabel=cms.string("discriminator")),
        byVVLooseIsolationMVArun2v1DBnewDMwLT = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLT"), provenanceConfigLabel=cms.string("workingPoints"), idLabel=cms.string("_VVLoose")),
        byVLooseIsolationMVArun2v1DBnewDMwLT = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLT"), provenanceConfigLabel=cms.string("workingPoints"), idLabel=cms.string("_VLoose")),
        byLooseIsolationMVArun2v1DBnewDMwLT = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLT"), provenanceConfigLabel=cms.string("workingPoints"), idLabel=cms.string("_Loose")),
        byMediumIsolationMVArun2v1DBnewDMwLT = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLT"), provenanceConfigLabel=cms.string("workingPoints"), idLabel=cms.string("_Medium")),
        byTightIsolationMVArun2v1DBnewDMwLT = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLT"), provenanceConfigLabel=cms.string("workingPoints"), idLabel=cms.string("_Tight")),
        byVTightIsolationMVArun2v1DBnewDMwLT = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLT"), provenanceConfigLabel=cms.string("workingPoints"), idLabel=cms.string("_VTight")),
        byVVTightIsolationMVArun2v1DBnewDMwLT = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLT"), provenanceConfigLabel=cms.string("workingPoints"), idLabel=cms.string("_VVTight")),
        chargedIsoPtSumdR03 = cms.PSet( inputTag = cms.InputTag("hpsPFTauBasicDiscriminatorsdR03"), provenanceConfigLabel=cms.string("IDdefinitions"), idLabel=cms.string("ChargedIsoPtSumdR03")),
        neutralIsoPtSumdR03 = cms.PSet( inputTag = cms.InputTag("hpsPFTauBasicDiscriminatorsdR03"), provenanceConfigLabel=cms.string("IDdefinitions"), idLabel=cms.string("NeutralIsoPtSumdR03")),
        neutralIsoPtSumWeightdR03 = cms.PSet( inputTag = cms.InputTag("hpsPFTauBasicDiscriminatorsdR03"), provenanceConfigLabel=cms.string("IDdefinitions"), idLabel=cms.string("NeutralIsoPtSumWeightdR03")),
        footprintCorrectiondR03 = cms.PSet( inputTag = cms.InputTag("hpsPFTauBasicDiscriminatorsdR03"), provenanceConfigLabel=cms.string("IDdefinitions"), idLabel=cms.string("TauFootprintCorrectiondR03")),
        photonPtSumOutsideSignalConedR03 = cms.PSet( inputTag = cms.InputTag("hpsPFTauBasicDiscriminatorsdR03"), provenanceConfigLabel=cms.string("IDdefinitions"), idLabel=cms.string("PhotonPtSumOutsideSignalConedR03")),
        byIsolationMVArun2v1DBdR03oldDMwLTraw = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLT"), provenanceConfigLabel=cms.string("rawValues"), idLabel=cms.string("discriminator")),
        byVVLooseIsolationMVArun2v1DBdR03oldDMwLT = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLT"), provenanceConfigLabel=cms.string("workingPoints"), idLabel=cms.string("_VVLoose")),
        byVLooseIsolationMVArun2v1DBdR03oldDMwLT = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLT"), provenanceConfigLabel=cms.string("workingPoints"), idLabel=cms.string("_VLoose")),
        byLooseIsolationMVArun2v1DBdR03oldDMwLT = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLT"), provenanceConfigLabel=cms.string("workingPoints"), idLabel=cms.string("_Loose")),
        byMediumIsolationMVArun2v1DBdR03oldDMwLT = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLT"), provenanceConfigLabel=cms.string("workingPoints"), idLabel=cms.string("_Medium")),
        byTightIsolationMVArun2v1DBdR03oldDMwLT = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLT"), provenanceConfigLabel=cms.string("workingPoints"), idLabel=cms.string("_Tight")),
        byVTightIsolationMVArun2v1DBdR03oldDMwLT = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLT"), provenanceConfigLabel=cms.string("workingPoints"), idLabel=cms.string("_VTight")),
        byVVTightIsolationMVArun2v1DBdR03oldDMwLT = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLT"), provenanceConfigLabel=cms.string("workingPoints"), idLabel=cms.string("_VVTight")),
        againstElectronMVA6Raw = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByMVA6ElectronRejection"), provenanceConfigLabel=cms.string("rawValues"), idLabel=cms.string("discriminator")),
        againstElectronMVA6category = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByMVA6ElectronRejection"), provenanceConfigLabel=cms.string("rawValues"), idLabel=cms.string("category")),
        againstElectronVLooseMVA6 = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByMVA6ElectronRejection"), provenanceConfigLabel=cms.string("workingPoints"), idLabel=cms.string("_VLoose")),
        againstElectronLooseMVA6 = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByMVA6ElectronRejection"), provenanceConfigLabel=cms.string("workingPoints"), idLabel=cms.string("_Loose")),
        againstElectronMediumMVA6 = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByMVA6ElectronRejection"), provenanceConfigLabel=cms.string("workingPoints"), idLabel=cms.string("_Medium")),
        againstElectronTightMVA6 = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByMVA6ElectronRejection"), provenanceConfigLabel=cms.string("workingPoints"), idLabel=cms.string("_Tight")),
        againstElectronVTightMVA6 = cms.PSet( inputTag = cms.InputTag("hpsPFTauDiscriminationByMVA6ElectronRejection"), provenanceConfigLabel=cms.string("workingPoints"), idLabel=cms.string("_VTight")),
    ),
    skipMissingTauID = cms.bool(False), #Allow to skip a tau ID variable when not present in the event"
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

