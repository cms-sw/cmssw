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
    tauIDSources = cms.PSet(),
    # IDs added below

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

# helper functions for ID configuration
def singleID(pset, inputID, label):
    setattr(pset, label, cms.PSet(inputTag=cms.InputTag(inputID), provenanceConfigLabel=cms.string(""), idLabel=cms.string("")))
def containerID(pset, inputID, provCfgLabel, wps):
    for wp in wps:
        setattr(pset, wp[0], cms.PSet(inputTag=cms.InputTag(inputID), provenanceConfigLabel=cms.string(provCfgLabel), idLabel=cms.string(wp[1])))

# configure many IDs
# you can comment out those you don't want to save some disk space
singleID(patTaus.tauIDSources, "hpsPFTauDiscriminationByDecayModeFinding", "decayModeFinding")
singleID(patTaus.tauIDSources, "hpsPFTauDiscriminationByDecayModeFindingNewDMs", "decayModeFindingNewDMs")
containerID(patTaus.tauIDSources, "hpsPFTauDiscriminationByMuonRejection3", "IDWPdefinitions", [
    ["againstMuonLoose3", "ByLooseMuonRejection3"],
    ["againstMuonTight3", "ByTightMuonRejection3"]
    ])
containerID(patTaus.tauIDSources, "hpsPFTauBasicDiscriminators", "IDdefinitions", [
    ["chargedIsoPtSum", "ChargedIsoPtSum"],
    ["neutralIsoPtSum", "NeutralIsoPtSum"],
    ["puCorrPtSum", "PUcorrPtSum"],
    ["neutralIsoPtSumWeight", "NeutralIsoPtSumWeight"],
    ["footprintCorrection", "TauFootprintCorrection"],
    ["photonPtSumOutsideSignalCone", "PhotonPtSumOutsideSignalCone"],
    ["byCombinedIsolationDeltaBetaCorrRaw3Hits", "ByRawCombinedIsolationDBSumPtCorr3Hits"]
    ])
containerID(patTaus.tauIDSources, "hpsPFTauBasicDiscriminators", "IDWPdefinitions", [
    ["byLooseCombinedIsolationDeltaBetaCorr3Hits", "ByLooseCombinedIsolationDBSumPtCorr3Hits"],
    ["byMediumCombinedIsolationDeltaBetaCorr3Hits", "ByMediumCombinedIsolationDBSumPtCorr3Hits"],
    ["byTightCombinedIsolationDeltaBetaCorr3Hits", "ByTightCombinedIsolationDBSumPtCorr3Hits"],
    ["byPhotonPtSumOutsideSignalCone", "ByPhotonPtSumOutsideSignalCone"]
    ])
containerID(patTaus.tauIDSources, "hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLT", "rawValues", [
    ["byIsolationMVArun2v1DBoldDMwLTraw", "discriminator"]
    ])
containerID(patTaus.tauIDSources, "hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLT", "workingPoints", [
    ["byVVLooseIsolationMVArun2v1DBoldDMwLT", "_VVLoose"],
    ["byVLooseIsolationMVArun2v1DBoldDMwLT", "_VLoose"],
    ["byLooseIsolationMVArun2v1DBoldDMwLT", "_Loose"],
    ["byMediumIsolationMVArun2v1DBoldDMwLT", "_Medium"],
    ["byTightIsolationMVArun2v1DBoldDMwLT", "_Tight"],
    ["byVTightIsolationMVArun2v1DBoldDMwLT", "_VTight"],
    ["byVVTightIsolationMVArun2v1DBoldDMwLT", "_VVTight"]
    ])
containerID(patTaus.tauIDSources, "hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLT", "rawValues", [
    ["byIsolationMVArun2v1DBnewDMwLTraw", "discriminator"]
    ])
containerID(patTaus.tauIDSources, "hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLT", "workingPoints", [
    ["byVVLooseIsolationMVArun2v1DBnewDMwLT", "_VVLoose"],
    ["byVLooseIsolationMVArun2v1DBnewDMwLT", "_VLoose"],
    ["byLooseIsolationMVArun2v1DBnewDMwLT", "_Loose"],
    ["byMediumIsolationMVArun2v1DBnewDMwLT", "_Medium"],
    ["byTightIsolationMVArun2v1DBnewDMwLT", "_Tight"],
    ["byVTightIsolationMVArun2v1DBnewDMwLT", "_VTight"],
    ["byVVTightIsolationMVArun2v1DBnewDMwLT", "_VVTight"]
    ])
containerID(patTaus.tauIDSources, "hpsPFTauBasicDiscriminatorsdR03", "IDdefinitions", [
    ["chargedIsoPtSumdR03", "ChargedIsoPtSumdR03"],
    ["neutralIsoPtSumdR03", "NeutralIsoPtSumdR03"],
    ["neutralIsoPtSumWeightdR03", "NeutralIsoPtSumWeightdR03"],
    ["footprintCorrectiondR03", "TauFootprintCorrectiondR03"],
    ["photonPtSumOutsideSignalConedR03", "PhotonPtSumOutsideSignalConedR03"]
    ])
containerID(patTaus.tauIDSources, "hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLT", "rawValues", [
    ["byIsolationMVArun2v1DBdR03oldDMwLTraw", "discriminator"]
    ])
containerID(patTaus.tauIDSources, "hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLT", "workingPoints", [
    ["byVVLooseIsolationMVArun2v1DBdR03oldDMwLT", "_VVLoose"],
    ["byVLooseIsolationMVArun2v1DBdR03oldDMwLT", "_VLoose"],
    ["byLooseIsolationMVArun2v1DBdR03oldDMwLT", "_Loose"],
    ["byMediumIsolationMVArun2v1DBdR03oldDMwLT", "_Medium"],
    ["byTightIsolationMVArun2v1DBdR03oldDMwLT", "_Tight"],
    ["byVTightIsolationMVArun2v1DBdR03oldDMwLT", "_VTight"],
    ["byVVTightIsolationMVArun2v1DBdR03oldDMwLT", "_VVTight"]
    ])    
containerID(patTaus.tauIDSources, "hpsPFTauDiscriminationByMVA6ElectronRejection", "rawValues", [
    ["againstElectronMVA6Raw", "discriminator"],
    ["againstElectronMVA6category", "category"]
    ])
containerID(patTaus.tauIDSources, "hpsPFTauDiscriminationByMVA6ElectronRejection", "workingPoints", [
    ["againstElectronVLooseMVA6", "_VLoose"],
    ["againstElectronLooseMVA6", "_Loose"],
    ["againstElectronMediumMVA6", "_Medium"],
    ["againstElectronTightMVA6", "_Tight"],
    ["againstElectronVTightMVA6", "_VTight"]
    ])
singleID(patTaus.tauIDSources, "hpsPFTauDiscriminationByDeadECALElectronRejection", "againstElectronDeadECAL")
