import FWCore.ParameterSet.Config as cms

particleFlowTmpBarrel = cms.EDProducer("PFProducer",
    GedElectronValueMap = cms.InputTag("gedGsfElectronValueMapsTmp"),
    GedPhotonValueMap = cms.InputTag("gedPhotonsTmp","valMapPFEgammaCandToPhoton"),
    PFEGammaCandidates = cms.InputTag("particleFlowEGamma"),
    PFEGammaFiltersParameters = cms.PSet(
        allowEEEinPF = cms.bool(False),
        electronDnnBkgThresholds = cms.PSet(
            electronDnnBkgHighPtBarrelThr = cms.double(0.8),
            electronDnnBkgHighPtEndcapThr = cms.double(0.75),
            electronDnnBkgLowPtThr = cms.double(0.75)
        ),
        electronDnnThresholds = cms.PSet(
            electronDnnHighPtBarrelThr = cms.double(0.068),
            electronDnnHighPtEndcapThr = cms.double(0.056),
            electronDnnLowPtThr = cms.double(0.075)
        ),
        electron_ecalDrivenHademPreselCut = cms.double(0.15),
        electron_iso_combIso_barrel = cms.double(10),
        electron_iso_combIso_endcap = cms.double(10),
        electron_iso_mva_barrel = cms.double(-0.1875),
        electron_iso_mva_endcap = cms.double(-0.1075),
        electron_iso_pt = cms.double(10),
        electron_maxElePtForOnlyMVAPresel = cms.double(50),
        electron_missinghits = cms.uint32(1),
        electron_noniso_mvaCut = cms.double(-0.1),
        electron_protectionsForBadHcal = cms.PSet(
            dEta = cms.vdouble(0.0064, 0.01264),
            dPhi = cms.vdouble(0.0547, 0.0394),
            eInvPInv = cms.vdouble(0.184, 0.0721),
            enableProtections = cms.bool(False),
            full5x5_sigmaIetaIeta = cms.vdouble(0.0106, 0.0387)
        ),
        electron_protectionsForJetMET = cms.PSet(
            maxDPhiIN = cms.double(0.1),
            maxE = cms.double(50),
            maxEcalEOverPRes = cms.double(0.2),
            maxEcalEOverP_1 = cms.double(0.5),
            maxEcalEOverP_2 = cms.double(0.2),
            maxEeleOverPout = cms.double(0.2),
            maxEeleOverPoutRes = cms.double(0.5),
            maxEleHcalEOverEcalE = cms.double(0.1),
            maxHcalE = cms.double(10),
            maxHcalEOverEcalE = cms.double(0.1),
            maxHcalEOverP = cms.double(1),
            maxNtracks = cms.double(3),
            maxTrackPOverEele = cms.double(1)
        ),
        photonDnnThresholds = cms.PSet(
            photonDnnBarrelThr = cms.double(0.22),
            photonDnnEndcapThr = cms.double(0.35)
        ),
        photon_HoE = cms.double(0.05),
        photon_MinEt = cms.double(10),
        photon_SigmaiEtaiEta_barrel = cms.double(0.0125),
        photon_SigmaiEtaiEta_endcap = cms.double(0.034),
        photon_combIso = cms.double(10),
        photon_protectionsForBadHcal = cms.PSet(
            enableProtections = cms.bool(False),
            solidConeTrkIsoOffset = cms.double(10),
            solidConeTrkIsoSlope = cms.double(0.3)
        ),
        photon_protectionsForJetMET = cms.PSet(
            sumPtTrackIso = cms.double(4),
            sumPtTrackIsoSlope = cms.double(0.001)
        ),
        useEBModelInGap = cms.bool(True),
        useElePFidDnn = cms.bool(False),
        usePhotonPFidDnn = cms.bool(False)
    ),
    PFHFCleaningParameters = cms.PSet(
        maxDeltaPhiPt = cms.double(7),
        maxSignificance = cms.double(2.5),
        minDeltaMet = cms.double(0.4),
        minHFCleaningPt = cms.double(5),
        minSignificance = cms.double(2.5),
        minSignificanceReduction = cms.double(1.4)
    ),
    PFMuonAlgoParameters = cms.PSet(
        cosmicRejectionDistance = cms.double(1),
        eventFactorForCosmics = cms.double(10),
        eventFractionForCleaning = cms.double(0.5),
        eventFractionForRejection = cms.double(0.8),
        maxDPtOPt = cms.double(1),
        metFactorForCleaning = cms.double(4),
        metFactorForFakes = cms.double(4),
        metFactorForHighEta = cms.double(25),
        metFactorForRejection = cms.double(4),
        metSignificanceForCleaning = cms.double(3),
        metSignificanceForRejection = cms.double(4),
        minEnergyForPunchThrough = cms.double(100),
        minMomentumForPunchThrough = cms.double(100),
        minPtForPostCleaning = cms.double(20),
        ptErrorScale = cms.double(8),
        ptFactorForHighEta = cms.double(2),
        punchThroughFactor = cms.double(3),
        punchThroughMETFactor = cms.double(4),
        trackQuality = cms.string('highPurity')
    ),
    blocks = cms.InputTag("particleFlowBlock"),
    calibHF_a_EMHAD = cms.vdouble(
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1
    ),
    calibHF_a_EMonly = cms.vdouble(
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1
    ),
    calibHF_b_EMHAD = cms.vdouble(
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1
    ),
    calibHF_b_HADonly = cms.vdouble(
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1
    ),
    calibHF_eta_step = cms.vdouble(
        0, 2.9, 3, 3.2, 4.2,
        4.4, 4.6, 4.8, 5.2, 5.4
    ),
    calibHF_use = cms.bool(False),
    calibrationsLabel = cms.string(''),
    cleanedHF = cms.VInputTag("particleFlowRecHitHF:Cleaned", "particleFlowClusterHF:Cleaned"),
    debug = cms.untracked.bool(False),
    dptRel_DispVtx = cms.double(10),
    egammaElectrons = cms.InputTag("mvaElectrons"),
    factors_45 = cms.vdouble(10, 100),
    goodPixelTrackDeadHcal_chi2n = cms.double(2),
    goodPixelTrackDeadHcal_dxy = cms.double(0.02),
    goodPixelTrackDeadHcal_dz = cms.double(0.05),
    goodPixelTrackDeadHcal_maxLost3Hit = cms.int32(0),
    goodPixelTrackDeadHcal_maxLost4Hit = cms.int32(1),
    goodPixelTrackDeadHcal_maxPt = cms.double(50),
    goodPixelTrackDeadHcal_minEta = cms.double(2.3),
    goodPixelTrackDeadHcal_ptErrRel = cms.double(1),
    goodTrackDeadHcal_chi2n = cms.double(5),
    goodTrackDeadHcal_dxy = cms.double(0.5),
    goodTrackDeadHcal_layers = cms.uint32(4),
    goodTrackDeadHcal_ptErrRel = cms.double(0.2),
    goodTrackDeadHcal_validFr = cms.double(0.5),
    iCfgCandConnector = cms.PSet(
        bCalibPrimary = cms.bool(True),
        bCorrect = cms.bool(True),
        dptRel_MergedTrack = cms.double(5),
        dptRel_PrimaryTrack = cms.double(10),
        nuclCalibFactors = cms.vdouble(0.8, 0.15, 0.5, 0.5, 0.05),
        ptErrorSecondary = cms.double(1)
    ),
    mightGet = cms.optional.untracked.vstring,
    muon_ECAL = cms.vdouble(0.5, 0.5),
    muon_HCAL = cms.vdouble(3, 3),
    muon_HO = cms.vdouble(0.9, 0.9),
    muons = cms.InputTag("hltPhase2L3Muons"),
    nsigma_TRACK = cms.double(1),
    pf_nsigma_ECAL = cms.double(0),
    pf_nsigma_HCAL = cms.double(1),
    pf_nsigma_HFEM = cms.double(1),
    pf_nsigma_HFHAD = cms.double(1),
    postHFCleaning = cms.bool(False),
    postMuonCleaning = cms.bool(True),
    pt_Error = cms.double(1),
    rejectTracks_Bad = cms.bool(True),
    rejectTracks_Step45 = cms.bool(True),
    resolHF_square = cms.vdouble(7.834401, 0.012996, 0),
    useCalibrationsFromDB = cms.bool(True),
    useEGammaElectrons = cms.bool(False),
    useEGammaFilters = cms.bool(False),
    useHO = cms.bool(True),
    usePFConversions = cms.bool(False),
    usePFDecays = cms.bool(False),
    usePFNuclearInteractions = cms.bool(False),
    useProtectionsForJetMET = cms.bool(False),
    useVerticesForNeutral = cms.bool(True),
    verbose = cms.untracked.bool(False),
    vertexCollection = cms.InputTag("offlinePrimaryVertices"),
    vetoEndcap = cms.bool(True)
)
