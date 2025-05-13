import FWCore.ParameterSet.Config as cms
from  PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.simpleCandidateFlatTableProducer_cfi import simpleCandidateFlatTableProducer
from Configuration.Eras.Modifier_run3_scouting_nanoAOD_2023_cff import run3_scouting_nanoAOD_2023
from Configuration.Eras.Modifier_run3_scouting_nanoAOD_2024_cff import run3_scouting_nanoAOD_2024

#####################################
##### Scouting Original Objects #####
#####################################
# objects directly from Run3Scouting* formats

# Scouting Muon
# https://github.com/cms-sw/cmssw/blob/CMSSW_14_0_X/DataFormats/Scouting/interface/Run3ScoutingMuon.h

scoutingMuonTable = cms.EDProducer("SimpleRun3ScoutingMuonCollectionFlatTableProducer",
    src = cms.InputTag("hltScoutingMuonPacker"),
    cut = cms.string(""),
    name = cms.string("ScoutingMuon"),
    doc  = cms.string("Scouting Muon"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = cms.PSet(
        pt = Var('pt', 'float', precision=10, doc='pt'),
        eta = Var('eta', 'float', precision=10, doc='eta'),
        phi = Var('phi', 'float', precision=10, doc='phi'),
        m = Var('m', 'float', precision=10, doc='mass'),
        type = Var('type', 'int', doc='type of muon'),
        charge = Var('charge', 'int', doc='track charge'),
        normchi2 = Var('normalizedChi2', 'float', precision=10, doc='normalized chi squared'),
        ecalIso = Var('ecalIso', 'float', precision=10, doc='PF ECAL isolation'),
        hcalIso = Var('hcalIso', 'float', precision=10, doc='PF HCAL isolation'),
        trackIso = Var('trackIso', 'float', precision=10, doc='track isolation'),
        nValidStandAloneMuonHits = Var('nValidStandAloneMuonHits', 'int', doc='number of valid standalone muon hits'),
        nStandAloneMuonMatchedStations = Var('nStandAloneMuonMatchedStations', 'int', doc='number of muon stations with valid hits'),
        nValidRecoMuonHits = Var('nValidRecoMuonHits', 'int', doc='number of valid reco muon hits'),
        nRecoMuonChambers = Var('nRecoMuonChambers', 'int', doc='number of reco muon chambers'),
        nRecoMuonChambersCSCorDT = Var('nRecoMuonChambersCSCorDT', 'int', doc='number of reco muon chambers CSC or DT'),
        nRecoMuonMatches = Var('nRecoMuonMatches', 'int', doc='number of reco muon matches'),
        nRecoMuonMatchedStations = Var('nRecoMuonMatchedStations', 'int', doc='number of reco muon matched stations'),
        nRecoMuonExpectedMatchedStations = Var('nRecoMuonExpectedMatchedStations', 'int', doc='number of reco muon expected matched stations'),
        recoMuonStationMask = Var('recoMuonStationMask', 'int', doc='reco muon station mask'),
        nRecoMuonMatchedRPCLayers = Var('nRecoMuonMatchedRPCLayers', 'int', doc='number of reco muon matched RPC layers'),
        recoMuonRPClayerMask = Var('recoMuonRPClayerMask', 'int', doc='reco muon RPC layer mask'),
        nValidPixelHits = Var('nValidPixelHits', 'int', doc='number of valid pixel hits'),
        nValidStripHits = Var('nValidStripHits', 'int', doc='number of valid strip hits'),
        nPixelLayersWithMeasurement = Var('nPixelLayersWithMeasurement', 'int', doc='number of pixel layers with measurement'),
        nTrackerLayersWithMeasurement = Var('nTrackerLayersWithMeasurement', 'int', doc='number of tracker layer with measurements'),
        trk_chi2 = Var('trk_chi2', 'float', precision=10, doc='track chi squared'),
        trk_ndof = Var('trk_ndof', 'float', precision=10, doc='track number of degrees of freedom'),
        trk_dxy = Var('trk_dxy', 'float', precision=10, doc='track dxy'),
        trk_dz = Var('trk_dz', 'float', precision=10, doc='track dz'),
        trk_qoverp = Var('trk_qoverp', 'float', precision=10, doc='track qoverp'),
        trk_lambda = Var('trk_lambda', 'float', precision=10, doc='track lambda'),
        trk_pt = Var('trk_pt', 'float', precision=10, doc='track pt'),
        trk_phi = Var('trk_phi', 'float', precision=10, doc='track phi'),
        trk_eta = Var('trk_eta', 'float', precision=10, doc='track eta'),
        trk_dxyError = Var('trk_dxyError', 'float', precision=10, doc='track dxyError'),
        trk_dzError = Var('trk_dzError', 'float', precision=10, doc='tracl dzError'),
        trk_qoverpError = Var('trk_qoverpError', 'float', precision=10, doc='track qoverpError'),
        trk_lambdaError = Var('trk_lambdaError', 'float', precision=10, doc='track lambdaError'),
        trk_phiError = Var('trk_phiError', 'float', precision=10, doc='track phiError'),
        trk_dsz = Var('trk_dsz', 'float', precision=10, doc='track dsz'),
        trk_dszError = Var('trk_dszError', 'float', precision=10, doc='track dszError'),
        trk_qoverp_lambda_cov = Var('trk_qoverp_lambda_cov', 'float', precision=10, doc='track qoverp lambda covariance ((0,1) element of covariance matrix)'),
        trk_qoverp_phi_cov = Var('trk_qoverp_phi_cov', 'float', precision=10, doc='track qoverp phi covariance ((0,2) element of covariance matrix)'),
        trk_qoverp_dxy_cov = Var('trk_qoverp_dxy_cov', 'float', precision=10, doc='track qoverp dxy covariance ((0,3) element of covariance matrix)'),
        trk_qoverp_dsz_cov = Var('trk_qoverp_dsz_cov', 'float', precision=10, doc='track qoverp dsz covariance ((0,4) element of covariance matrix)'),
        trk_lambda_phi_cov = Var('trk_lambda_phi_cov', 'float', precision=10, doc='track lambda phi covariance ((1,2) element of covariance matrix)'),
        trk_lambda_dxy_cov = Var('trk_lambda_dxy_cov', 'float', precision=10, doc='track lambda dxy covariance ((1,3) element of covariance matrix)'),
        trk_lambda_dsz_cov = Var('trk_lambda_dsz_cov', 'float', precision=10, doc='track lambda dsz covariance ((1,4) element of covariance matrix)'),
        trk_phi_dxy_cov = Var('trk_phi_dxy_cov', 'float', precision=10, doc='track phi dxy covariance ((2,3) element of covariance matrix)'),
        trk_phi_dsz_cov = Var('trk_phi_dsz_cov', 'float', precision=10, doc='track phi dsz covariance ((2,4) element of covariance matrix)'),
        trk_dxy_dsz_cov = Var('trk_dxy_dsz_cov', 'float', precision=10, doc='track dxy dsz covariance ((3,4) element of covariance matrix)'),
        trk_vx = Var('trk_vx', 'float', precision=10, doc='track vx'),
        trk_vy = Var('trk_vy', 'float', precision=10, doc='track vy'),
        trk_vz = Var('trk_vz', 'float', precision=10, doc='track vz'),
        trk_hitPattern_hitCount = Var("trk_hitPattern().hitCount", "uint8", doc="track hitPattern hitCount"),
        trk_hitPattern_beginTrackHits = Var("trk_hitPattern().beginTrackHits", "uint8", doc="track hitPattern beginTrackHits"),
        trk_hitPattern_endTrackHits = Var("trk_hitPattern().endTrackHits", "uint8", doc="track hitPattern endTrackHits"),
        trk_hitPattern_beginInner = Var("trk_hitPattern().beginInner", "uint8", doc="track hitPattern beginInner"),
        trk_hitPattern_endInner = Var("trk_hitPattern().endInner", "uint8", doc="track hitPattern endInner"),
        trk_hitPattern_beginOuter = Var("trk_hitPattern().beginOuter", "uint8", doc="track hitPattern beginOuter"),
        trk_hitPattern_endOuter = Var("trk_hitPattern().endOuter", "uint8", doc="track hitPattern endOuter"),
    ),
    collectionVariables = cms.PSet(
        ScoutingMuonVtxIndx = cms.PSet(
            name = cms.string("ScoutingMuonVtxIndx"),
            doc = cms.string("Scouting Muon Displaced Vertex Index"),
            useCount = cms.bool(True),
            useOffset = cms.bool(True),
            variables = cms.PSet(
                vtxIndx = Var('vtxIndx', 'int', doc='vertex indices'),
            ),
        ),
        ScoutingMuonHitPattern = cms.PSet(
            name = cms.string("ScoutingMuonHitPattern"),
            doc = cms.string("Scouting Muon HitPattern"),
            useCount = cms.bool(True),
            useOffset = cms.bool(True),
            variables = cms.PSet(
                hitPattern = Var('trk_hitPattern().hitPattern', 'uint16', doc='track hitPattern hitPattern'),
            ),
        )
    )
)

# Scouting Vertex
# format during 2022-23 data-taking used for both primary vertex and dimuon displaced vertex
# https://github.com/cms-sw/cmssw/blob/CMSSW_14_0_X/DataFormats/Scouting/interface/Run3ScoutingVertex.h

scoutingVertexVariables = cms.PSet(
    x = Var('x', 'float', precision=10, doc='position x coordinate'),
    y = Var('y', 'float', precision=10, doc='position y coordinate'),
    z = Var('z', 'float', precision=10, doc='position z coordinate'),
    xError = Var('xError', 'float', precision=10, doc='x error'),
    yError = Var('yError', 'float', precision=10, doc='y error'),
    zError = Var('zError', 'float', precision=10, doc='z error'),
    tracksSize = Var('tracksSize', 'int', doc='number of tracks'),
    chi2 = Var('chi2', 'float', precision=10, doc='chi squared'),
    ndof = Var('ndof', 'int', doc='number of degrees of freedom'),
    isValidVtx = Var('isValidVtx', 'bool', doc='is valid'),
)

# scouting vertex format changed for 2024 data-taking in https://github.com/cms-sw/cmssw/pull/43758
# used for both primary vertex and dimuon displaced vertex
# https://github.com/cms-sw/cmssw/blob/CMSSW_14_0_X/DataFormats/Scouting/interface/Run3ScoutingVertex.h

run3_scouting_nanoAOD_2024.toModify(
    scoutingVertexVariables,
    xyCov = Var('xyCov', 'float', precision=10, doc='xy covariance'),
    xzCov = Var('xzCov', 'float', precision=10, doc='xz covariance'),
    yzCov = Var('yzCov', 'float', precision=10, doc='yz covariance'),
)

# Scouting Displaced Vertex (from dimuon)
# https://github.com/cms-sw/cmssw/blob/CMSSW_14_0_X/DataFormats/Scouting/interface/Run3ScoutingVertex.h

scoutingMuonDisplacedVertexTable = cms.EDProducer("SimpleRun3ScoutingVertexFlatTableProducer",
    src = cms.InputTag("hltScoutingMuonPacker","displacedVtx"),
    cut = cms.string(""),
    name = cms.string("ScoutingMuonDisplacedVertex"),
    doc  = cms.string("Scouting Muon Displaced Vertex"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = scoutingVertexVariables,
)

# from 2024, there are two scouting muon collections

# muonVtx
scoutingMuonVtxTable = scoutingMuonTable.clone(
    src = cms.InputTag("hltScoutingMuonPackerVtx"),
    name = cms.string("ScoutingMuonVtx"),
    doc  = cms.string("Scouting Muon Vtx"),
)
scoutingMuonVtxTable.collectionVariables.ScoutingMuonVtxIndx.name = cms.string("ScoutingMuonVtxVtxIndx")
scoutingMuonVtxTable.collectionVariables.ScoutingMuonVtxIndx.doc = cms.string("ScoutingMuonVtx VtxIndx")
scoutingMuonVtxTable.collectionVariables.ScoutingMuonHitPattern.name = cms.string("ScoutingMuonVtxHitPattern")
scoutingMuonVtxTable.collectionVariables.ScoutingMuonHitPattern.doc = cms.string("ScoutingMuonVtx HitPattern")

scoutingMuonVtxDisplacedVertexTable = scoutingMuonDisplacedVertexTable.clone(
    src = cms.InputTag("hltScoutingMuonPackerVtx", "displacedVtx"),
    name = cms.string("ScoutingMuonVtxDisplacedVertex"),
    doc  = cms.string("Scouting Muon Vtx DisplacedVertex"),
)

# muonNoVtx
scoutingMuonNoVtxTable = scoutingMuonTable.clone(
    src = cms.InputTag("hltScoutingMuonPackerNoVtx"),
    name = cms.string("ScoutingMuonNoVtx"),
    doc  = cms.string("Scouting Muon NoVtx"),
)
scoutingMuonNoVtxTable.collectionVariables.ScoutingMuonVtxIndx.name = cms.string("ScoutingMuonNoVtxVtxIndx")
scoutingMuonNoVtxTable.collectionVariables.ScoutingMuonVtxIndx.doc = cms.string("ScoutingMuonNoVtx VtxIndx")
scoutingMuonNoVtxTable.collectionVariables.ScoutingMuonHitPattern.name = cms.string("ScoutingMuonNoVtxHitPattern")
scoutingMuonNoVtxTable.collectionVariables.ScoutingMuonHitPattern.doc = cms.string("ScoutingMuonNoVtx HitPattern")

scoutingMuonNoVtxDisplacedVertexTable = scoutingMuonDisplacedVertexTable.clone(
    src = cms.InputTag("hltScoutingMuonPackerNoVtx", "displacedVtx"),
    name = cms.string("ScoutingMuonNoVtxDisplacedVertex"),
    doc  = cms.string("Scouting Muon NoVtx DisplacedVertex"),
)

# Scouting Electron
# format during 2022 data-taking
# for accessing d0, dz, and charge, use changes from https://github.com/cms-sw/cmssw/pull/41025
# https://github.com/cms-sw/cmssw/blob/CMSSW_12_4_X/DataFormats/Scouting/interface/Run3ScoutingElectron.h

scoutingElectronTable = cms.EDProducer("SimpleRun3ScoutingElectronCollectionFlatTableProducer",
    src = cms.InputTag("hltScoutingEgammaPacker"),
    cut = cms.string(""),
    name = cms.string("ScoutingElectron"),
    doc  = cms.string("Scouting Electron"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = cms.PSet(
        pt = Var('pt', 'float', precision=10, doc='super-cluster (SC) pt'),
        eta = Var('eta', 'float', precision=10, doc='super-cluster (SC) eta'),
        phi = Var('phi', 'float', precision=10, doc='super-cluster (SC) phi'),
        m = Var('m', 'float', precision=10, doc='super-cluster (SC) mass'),
        d0 = Var('trkd0[0]', 'float', precision=10, doc='track d0'),
        dz = Var('trkdz[0]', 'float', precision=10, doc='track dz'),
        dEtaIn = Var('dEtaIn', 'float', precision=10, doc='#Delta#eta(SC seed, track pixel seed)'),
        dPhiIn = Var('dPhiIn', 'float', precision=10, doc='#Delta#phi(SC seed, track pixel seed)'),
        sigmaIetaIeta = Var('sigmaIetaIeta', 'float', precision=10, doc='sigmaIetaIeta of the SC, calculated with full 5x5 region, noise cleaned'),
        hOverE = Var('hOverE', 'float', precision=10, doc='Energy in HCAL / Energy in ECAL'),
        ooEMOop = Var('ooEMOop', 'float', precision=10, doc='1/E(SC) - 1/p(track momentum)'),
        missingHits = Var('missingHits', 'int', doc='missing hits in the tracker'),
        charge = Var('trkcharge[0]', 'int', doc='track charge'),
        ecalIso = Var('ecalIso', 'float', precision=10, doc='Isolation of SC in the ECAL'),
        hcalIso = Var('hcalIso', 'float', precision=10, doc='Isolation of SC in the HCAL'),
        trackIso = Var('trackIso', 'float', precision=10, doc='Isolation of electron track in the tracker'),
        r9 = Var('r9', 'float', precision=10, doc='Electron SC r9 as defined in https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideEgammaShowerShape'),
        sMin = Var('sMin', 'float', precision=10, doc='minor moment of the SC shower shape'),
        sMaj = Var('sMaj', 'float', precision=10, doc='major moment of the SC shower shape'),
        seedId = Var('seedId', 'uint', doc='ECAL ID of the SC seed'),
        rechitZeroSuppression = Var('rechitZeroSuppression', 'bool', doc='rechit zero suppression'),
    ),
    externalVariables = cms.PSet(),
)

# scouting electron format changed for 2023 data-taking in https://github.com/cms-sw/cmssw/pull/41025
# https://github.com/cms-sw/cmssw/blob/CMSSW_13_0_X/DataFormats/Scouting/interface/Run3ScoutingElectron.h

# from 2023, scouting electron's tracks are added as std::vector since multiple tracks are associated to scouting electrons
# this plugin selects the best track to reduce to a single track per scouting electron, which is more suitable for NanoAOD format
# https://github.com/cms-sw/cmssw/pull/47726

scoutingElectronBestTrack = cms.EDProducer("Run3ScoutingElectronBestTrackProducer",
    Run3ScoutingElectron = cms.InputTag("hltScoutingEgammaPacker"),
    TrackPtMin = cms.vdouble(12.0, 12.0),
    TrackChi2OverNdofMax = cms.vdouble(3.0, 2.0),
    RelativeEnergyDifferenceMax = cms.vdouble(1.0, 1.0),
    DeltaPhiMax = cms.vdouble(0.06, 0.06)
)

(run3_scouting_nanoAOD_2023 | run3_scouting_nanoAOD_2024).toModify(
    scoutingElectronTable.variables,
    d0 = None,      # replaced with trkd0 (std::vector)
    dz = None,      # replaced with trkdz (std::vector)
    charge = None,  # replaced with trkcharge (std::vector)
).toModify(
    scoutingElectronTable.externalVariables,
    bestTrack_d0 = ExtVar(cms.InputTag("scoutingElectronBestTrack", "Run3ScoutingElectronTrackd0"), float, doc="best track d0"),
    bestTrack_dz = ExtVar(cms.InputTag("scoutingElectronBestTrack", "Run3ScoutingElectronTrackdz"), float, doc="best track dz"),
    bestTrack_pt = ExtVar(cms.InputTag("scoutingElectronBestTrack", "Run3ScoutingElectronTrackpt"), float, doc="best track pt"),
    bestTrack_eta = ExtVar(cms.InputTag("scoutingElectronBestTrack", "Run3ScoutingElectronTracketa"), float, doc="best track eta"),
    bestTrack_phi = ExtVar(cms.InputTag("scoutingElectronBestTrack", "Run3ScoutingElectronTrackphi"), float, doc="best track phi"),
    bestTrack_chi2overndf = ExtVar(cms.InputTag("scoutingElectronBestTrack", "Run3ScoutingElectronTrackchi2overndf"), float, doc="best track chi2overndf"),
    bestTrack_charge = ExtVar(cms.InputTag("scoutingElectronBestTrack", "Run3ScoutingElectronTrackcharge"), int, doc="best track charge"),
)

# scouting electron format changed for 2024 data-taking in https://github.com/cms-sw/cmssw/pull/43744
# https://github.com/cms-sw/cmssw/blob/CMSSW_14_0_X/DataFormats/Scouting/interface/Run3ScoutingElectron.h

run3_scouting_nanoAOD_2024.toModify(
    scoutingElectronTable.variables,
    rawEnergy = Var("rawEnergy", "float", precision=10, doc="raw energy"),
    preshowerEnergy = Var("preshowerEnergy", "float", precision=10, doc='preshower energy'),
    corrEcalEnergyError = Var("corrEcalEnergyError", "float", precision=10, doc='corrEcalEnergyError'),
    trackfbrem = Var("trackfbrem", "float", precision=10, doc="trackfbrem"),
    nClusters = Var("nClusters", "uint", precision=10, doc="number of clusters"),
    nCrystals = Var("nCrystals", "uint", precision=10, doc="number of crystals"),
).toModify(
    scoutingElectronTable.externalVariables,
    bestTrack_pMode = ExtVar(cms.InputTag("scoutingElectronBestTrack", "Run3ScoutingElectronTrackpMode"), float, doc="best track pMode"),
    bestTrack_etaMode = ExtVar(cms.InputTag("scoutingElectronBestTrack", "Run3ScoutingElectronTracketaMode"), float, doc="best track etaMode"),
    bestTrack_phiMode = ExtVar(cms.InputTag("scoutingElectronBestTrack", "Run3ScoutingElectronTrackphiMode"), float, doc="best track phiMode"),
    bestTrack_qoverpModeError = ExtVar(cms.InputTag("scoutingElectronBestTrack", "Run3ScoutingElectronTrackqoverpModeError"), float, doc="best track qoverpModeError"),
)

# Scouting Photon
# format during 2022-23 data-taking
# https://github.com/cms-sw/cmssw/blob/CMSSW_13_0_X/DataFormats/Scouting/interface/Run3ScoutingPhoton.h

scoutingPhotonTable = cms.EDProducer("SimpleRun3ScoutingPhotonFlatTableProducer",
    src = cms.InputTag("hltScoutingEgammaPacker"),
    cut = cms.string(""),
    name = cms.string("ScoutingPhoton"),
    doc  = cms.string("Scouting Photon"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = cms.PSet(
        pt = Var('pt', 'float', precision=10, doc='super-cluster (SC) pt'),
        eta = Var('eta', 'float', precision=10, doc='SC eta'),
        phi = Var('phi', 'float', precision=10, doc='SC phi'),
        m = Var('m', 'float', precision=10, doc='SC mass'),
        sigmaIetaIeta = Var('sigmaIetaIeta', 'float', precision=10, doc='sigmaIetaIeta of the SC, calculated with full 5x5 region, noise cleaned'),
        hOverE = Var('hOverE', 'float', precision=10, doc='Energy in HCAL / Energy in ECAL'),
        ecalIso = Var('ecalIso', 'float', precision=10, doc='Isolation of SC in the ECAL'),
        hcalIso = Var('hcalIso', 'float', precision=10, doc='Isolation of SC in the HCAL'),
        trkIso = Var('trkIso', 'float', precision=10, doc='Isolation of track in the tracker'),
        r9 = Var('r9', 'float', precision=10, doc='Photon SC r9 as defined in https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideEgammaShowerShape'),
        sMin = Var('sMin', 'float', precision=10, doc='minor moment of the SC shower shape'),
        sMaj = Var('sMaj', 'float', precision=10, doc='major moment of the SC shower shape'),
        seedId = Var('seedId', 'uint', doc='ECAL ID of the SC seed'),
        rechitZeroSuppression = Var('rechitZeroSuppression', 'bool', doc='rechit zero suppression'),
    ),
)

# scouting photon format changed for 2024 data-taking in https://github.com/cms-sw/cmssw/pull/43744
# https://github.com/cms-sw/cmssw/blob/CMSSW_14_0_X/DataFormats/Scouting/interface/Run3ScoutingPhoton.h

run3_scouting_nanoAOD_2024.toModify(
    scoutingPhotonTable.variables,
    rawEnergy = Var("rawEnergy", "float", precision=10, doc="raw energy"),
    preshowerEnergy = Var("preshowerEnergy", "float", precision=10, doc='preshower energy'),
    corrEcalEnergyError = Var("corrEcalEnergyError", "float", precision=10, doc='corrEcalEnergyError'),
    nClusters = Var("nClusters", "uint", precision=10, doc="number of clusters"),
    nCrystals = Var("nCrystals", "uint", precision=10, doc="number of crystals"),
)

# Scouting Track
# https://github.com/cms-sw/cmssw/blob/CMSSW_14_0_X/DataFormats/Scouting/interface/Run3ScoutingTrack.h

scoutingTrackTable = cms.EDProducer("SimpleRun3ScoutingTrackFlatTableProducer",
    src = cms.InputTag("hltScoutingTrackPacker"),
    cut = cms.string(""),
    name = cms.string("ScoutingTrack"),
    doc  = cms.string("Scouting Track"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = cms.PSet(
        pt = Var('tk_pt', 'float', precision=10, doc='pt'),
        eta = Var('tk_eta', 'float', precision=10, doc='eta'),
        phi = Var('tk_phi', 'float', precision=10, doc='phi'),
        chi2 = Var('tk_chi2', 'float', precision=10, doc='chi squared'),
        ndof = Var('tk_ndof', 'float', precision=10, doc='number of degrees of freedom'),
        charge = Var('tk_charge', 'int', doc='charge'),
        dxy = Var('tk_dxy', 'float', precision=10, doc='dxy'),
        dz = Var('tk_dz', 'float', precision=10, doc='dz'),
        nValidPixelHits = Var('tk_nValidPixelHits', 'int', doc='number of valid pixel hits'),
        nValidStripHits = Var('tk_nValidStripHits', 'int', doc='number of valid strip hits'),
        nTrackerLayersWithMeasurement = Var('tk_nTrackerLayersWithMeasurement', 'int', doc='number of tracker layers with measurements'),
        qoverp = Var('tk_qoverp', 'float', precision=10, doc='qoverp'),
        lambda_ = Var('tk_lambda', 'float', precision=10, doc='lambda'),
        dxyError = Var('tk_dxy_Error', 'float', precision=10, doc='dxyError'),
        dzError = Var('tk_dz_Error', 'float', precision=10, doc='dzError'),
        qoverpError = Var('tk_qoverp_Error', 'float', precision=10, doc='qoverpError'),
        lambdaError = Var('tk_lambda_Error', 'float', precision=10, doc='lambdaError'),
        phiError = Var('tk_phi_Error', 'float', precision=10, doc='phiError'),
        dsz = Var('tk_dsz', 'float', precision=10, doc='dsz'),
        dszError = Var('tk_dsz_Error', 'float', precision=10, doc='dszError'),
        qoverp_lambda_cov = Var('tk_qoverp_lambda_cov', 'float', precision=10, doc='qoverp lambda covariance ((0,1) element of covariance matrix)'),
        qoverp_phi_cov = Var('tk_qoverp_phi_cov', 'float', precision=10, doc='qoverp phi covariance ((0,2) element of covariance matrix)'),
        qoverp_dxy_cov = Var('tk_qoverp_dxy_cov', 'float', precision=10, doc='qoverp dxy covariance ((0,3) element of covariance matrix)'),
        qoverp_dsz_cov = Var('tk_qoverp_dsz_cov', 'float', precision=10, doc='qoverp dsz covariance ((0,4) element of covariance matrix)'),
        lambda_phi_cov = Var('tk_lambda_phi_cov', 'float', precision=10, doc='lambda phi covariance ((1,2) element of covariance matrix)'),
        lambda_dxy_cov = Var('tk_lambda_dxy_cov', 'float', precision=10, doc='lambda dxy covariance ((1,3) element of covariance matrix)'),
        lambda_dsz_cov = Var('tk_lambda_dsz_cov', 'float', precision=10, doc='lambd dsz covariance ((1,4) element of covariance matrix)'),
        phi_dxy_cov = Var('tk_phi_dxy_cov', 'float', precision=10, doc='phi dxy covariance ((2,3) element of covariance matrix)'),
        phi_dsz_cov = Var('tk_phi_dsz_cov', 'float', precision=10, doc='phi dsz covariance ((2,4) element of covariance matrix)'),
        dxy_dsz_cov = Var('tk_dxy_dsz_cov', 'float', precision=10, doc='dxy dsz covariance ((3,4) element of covariance matrix)'),
        vtxInd = Var('tk_vtxInd', 'int', doc='vertex index'),
        vx = Var('tk_vx', 'float', precision=10, doc='vx'),
        vy = Var('tk_vy', 'float', precision=10, doc='vy'),
        vz = Var('tk_vz', 'float', precision=10, doc='vz'),
    ),
)

# Scouting Primary Vertex
# https://github.com/cms-sw/cmssw/blob/CMSSW_14_0_X/DataFormats/Scouting/interface/Run3ScoutingVertex.h

scoutingPrimaryVertexTable = cms.EDProducer("SimpleRun3ScoutingVertexFlatTableProducer",
    src = cms.InputTag("hltScoutingPrimaryVertexPacker", "primaryVtx"),
    cut = cms.string(""),
    name = cms.string("ScoutingPrimaryVertex"),
    doc  = cms.string("Scouting Primary Vertex"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = scoutingVertexVariables,
)

# Scouting Particle (PF candidate)
# https://github.com/cms-sw/cmssw/blob/CMSSW_14_0_X/DataFormats/Scouting/interface/Run3ScoutingParticle.h

scoutingParticleTable = cms.EDProducer("SimpleRun3ScoutingParticleFlatTableProducer",
    src = cms.InputTag("hltScoutingPFPacker"),
    name = cms.string("ScoutingParticle"),
    cut = cms.string(""),
    doc = cms.string("Scouting Particle"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = cms.PSet(
        P3Vars,
        pdgId = Var("pdgId", int, doc="PDG code assigned by the event reconstruction (not by MC truth)"),
        vertex = Var("vertex()", int, doc="vertex index"),
        normchi2 = Var("normchi2()", float, doc="normalized chi squared of best track"),
        dz = Var("dz()", float, doc="dz of best track"),
        dxy = Var("dxy()", float, doc="dxy of best track"),
        dzsig = Var("dzsig()", float, doc="dzsig of best track"),
        dxysig = Var("dxysig()", float, doc="dxysig of best track"),
        lostInnerHits = Var("lostInnerHits()", "uint8", doc="lostInnerHits of best track"),
        quality = Var("quality()", "uint8", doc="quality of best track"),
        trk_pt = Var("trk_pt()", "float", doc="pt of best track"),
        trk_eta = Var("trk_eta()", "float", doc="eta of best track"),
        trk_phi = Var("trk_phi()", "float", doc="phi of best track"),
        relative_trk_vars = Var("relative_trk_vars()", "bool", doc="relative_trk_vars"),
    ),
)

# Scouting PFJet
# https://github.com/cms-sw/cmssw/blob/CMSSW_14_0_X/DataFormats/Scouting/interface/Run3ScoutingPFJet.h

scoutingPFJetTable = cms.EDProducer("SimpleRun3ScoutingPFJetFlatTableProducer",
    src = cms.InputTag("hltScoutingPFPacker"),
    cut = cms.string(""),
    name = cms.string("ScoutingPFJet"),
    doc  = cms.string("Scouting PFJet"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = cms.PSet(
        P3Vars,
        m = Var('m', 'float', precision=10, doc='mass'),
        jetArea = Var('jetArea', 'float', precision=10, doc='jet area'),
        chargedHadronEnergy = Var('chargedHadronEnergy', 'float', precision=10, doc='charged hadron energy'),
        neutralHadronEnergy = Var('neutralHadronEnergy', 'float', precision=10, doc='neutral hadron energy'),
        photonEnergy = Var('photonEnergy', 'float', precision=10, doc='photon energy'),
        electronEnergy = Var('electronEnergy', 'float', precision=10, doc='electron energy'),
        muonEnergy = Var('muonEnergy', 'float', precision=10, doc='muon energy'),
        HFHadronEnergy = Var('HFHadronEnergy', 'float', precision=10, doc='hadronic energy in HF'),
        HFEMEnergy = Var('HFEMEnergy', 'float', precision=10, doc='electromagnetic energy in HF'),
        chargedHadronMultiplicity = Var('chargedHadronMultiplicity', 'int', doc='number of charged hadrons in the jet'),
        neutralHadronMultiplicity = Var('neutralHadronMultiplicity', 'int', doc='number of neutral hadrons in the jet'),
        photonMultiplicity = Var('photonMultiplicity', 'int', doc='number of photons in the jet'),
        electronMultiplicity = Var('electronMultiplicity', 'int', doc='number of electrons in the jet'),
        muonMultiplicity = Var('muonMultiplicity', 'int', doc='number of muons in the jet'),
        HFHadronMultiplicity = Var('HFHadronMultiplicity', 'int', doc='number of hadronic particles in the jet in HF'),
        HFEMMultiplicity = Var('HFEMMultiplicity', 'int', doc='number of electromagnetic particles in the jet in HF'),
        HOEnergy = Var('HOEnergy', 'float', precision=10, doc='hadronic energy in HO'),
    ),
)

# Scouting MET
scoutingMETTable = cms.EDProducer("GlobalVariablesTableProducer",
    name = cms.string("ScoutingMET"),
    variables = cms.PSet(
        pt = ExtVar( cms.InputTag("hltScoutingPFPacker", "pfMetPt"), "double", doc = "pt"),
        phi = ExtVar( cms.InputTag("hltScoutingPFPacker", "pfMetPhi"), "double", doc = "phi"),
    ),
)

# Scouting Rho
scoutingRhoTable = cms.EDProducer("GlobalVariablesTableProducer",
    name = cms.string("ScoutingRho"),
    variables = cms.PSet(
        fixedGridRhoFastjetAll = ExtVar(cms.InputTag("hltScoutingPFPacker", "rho"), "double", doc = "rho from all scouting PF Candidates, used e.g. for JECs" ),
    ),
)

####################################
##### Scouting Derived Objects #####
####################################
# objects built from scouting objects, e.g. reclustered jets

#########################
# Scouting PF Candidate #
#########################
# translation from Run3ScoutingParticle to reco::PFCandidate
# used as input for standard algorithm, e.g. jet clustering

scoutingPFCandidate = cms.EDProducer("Run3ScoutingParticleToRecoPFCandidateProducer",
    scoutingparticle = cms.InputTag("hltScoutingPFPacker"),
    CHS = cms.bool(False),
)

# this table is similar to scoutingParticleTable
# except if relative_trk_vars is true, PF candidate variables will be already added to PF candidate's track variables
scoutingPFCandidateTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("scoutingPFCandidate"),
    name = cms.string("ScoutingPFCandidate"),
    cut = cms.string(""),
    doc = cms.string("Scouting Candidate"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = cms.PSet(
        CandVars,
    ),
    externalVariables = cms.PSet(
        vertexIndex = ExtVar(cms.InputTag("scoutingPFCandidate", "vertexIndex"), int, doc="vertex index"),
        trkNormchi2 = ExtVar(cms.InputTag("scoutingPFCandidate", "normchi2"), float, doc="normalized chi squared of best track", precision=6),
        trkDz = ExtVar(cms.InputTag("scoutingPFCandidate", "dz"), float, doc="dz of best track", precision=6),
        trkDxy = ExtVar(cms.InputTag("scoutingPFCandidate", "dxy"), float, doc="dxy of best track", precision=6),
        trkDzsig = ExtVar(cms.InputTag("scoutingPFCandidate", "dzsig"), float, doc="dzsig of best track", precision=6),
        trkDxysig = ExtVar(cms.InputTag("scoutingPFCandidate", "dxysig"), float, doc="dxysig of best track", precision=6),
        trkLostInnerHits = ExtVar(cms.InputTag("scoutingPFCandidate", "lostInnerHits"), int, doc="lostInnerHits of best track"),
        trkQuality = ExtVar(cms.InputTag("scoutingPFCandidate", "quality"), int, doc="quality of best track"),
        trkPt = ExtVar(cms.InputTag("scoutingPFCandidate", "trkPt"), float, doc="pt of best track", precision=6),
        trkEta = ExtVar(cms.InputTag("scoutingPFCandidate", "trkEta"), float, doc="eta of best track", precision=6),
        trkPhi = ExtVar(cms.InputTag("scoutingPFCandidate", "trkPhi"), float, doc="phi of best track", precision=6),
    ),
)

#########################
# AK4 PFJet Reclustered #
#########################
# AK4 jets from reclustering PF candidates

# AK4 jet clustering

from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJets
scoutingPFJetRecluster = ak4PFJets.clone(
    src = ("scoutingPFCandidate"),
    jetPtMin = 20,
)

# AK4 jet tagging 

scoutingPFJetReclusterParticleNetJetTagInfos = cms.EDProducer("DeepBoostedJetTagInfoProducer",
    jet_radius = cms.double(0.4),
    min_jet_pt = cms.double(5.0),
    max_jet_eta = cms.double(2.5),
    min_pt_for_track_properties = cms.double(0.95),
    min_pt_for_pfcandidates = cms.double(0.1),
    use_puppiP4 = cms.bool(False),
    include_neutrals = cms.bool(True),
    sort_by_sip2dsig = cms.bool(False),
    min_puppi_wgt = cms.double(-1.0),
    flip_ip_sign = cms.bool(False),
    sip3dSigMax = cms.double(-1.0),
    use_hlt_features = cms.bool(False),
    pf_candidates = cms.InputTag("scoutingPFCandidate"),
    jets = cms.InputTag("scoutingPFJetRecluster"),
    puppi_value_map = cms.InputTag(""),
    use_scouting_features = cms.bool(True),
    normchi2_value_map = cms.InputTag("scoutingPFCandidate", "normchi2"),
    dz_value_map = cms.InputTag("scoutingPFCandidate", "dz"),
    dxy_value_map = cms.InputTag("scoutingPFCandidate", "dxy"),
    dzsig_value_map = cms.InputTag("scoutingPFCandidate", "dzsig"),
    dxysig_value_map = cms.InputTag("scoutingPFCandidate", "dxysig"),
    lostInnerHits_value_map = cms.InputTag("scoutingPFCandidate", "lostInnerHits"),
    quality_value_map = cms.InputTag("scoutingPFCandidate", "quality"),
    trkPt_value_map = cms.InputTag("scoutingPFCandidate", "trkPt"),
    trkEta_value_map = cms.InputTag("scoutingPFCandidate", "trkEta"),
    trkPhi_value_map = cms.InputTag("scoutingPFCandidate", "trkPhi"),
)

from RecoBTag.ONNXRuntime.boostedJetONNXJetTagsProducer_cfi import boostedJetONNXJetTagsProducer
scoutingPFJetReclusterParticleNetJetTags = cms.EDProducer("BoostedJetONNXJetTagsProducer",
    jets = cms.InputTag("scoutingPFJetRecluster"),
    produceValueMap = cms.untracked.bool(True),
    src = cms.InputTag("scoutingPFJetReclusterParticleNetJetTagInfos"),
    preprocess_json = cms.string("RecoBTag/Combined/data/Run3Scouting/ParticleNetAK4/V00/preprocess.json"),
    model_path = cms.FileInPath("RecoBTag/Combined/data/Run3Scouting/ParticleNetAK4/V00/particle-net.onnx"),
    flav_names = cms.vstring(["probb", "probbb","probc", "probcc", "probuds", "probg", "probundef"]),
    debugMode = cms.untracked.bool(False),
)

# output AK4 jet to nanoaod::flattable

scoutingPFJetReclusterTable = cms.EDProducer("SimplePFJetFlatTableProducer",
    src = cms.InputTag("scoutingPFJetRecluster"),
    name = cms.string("ScoutingPFJetRecluster"),
    cut = cms.string(""),
    doc = cms.string("ak4 jet from reclustering scouting PF candidates"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = cms.PSet(
        P4Vars,
        area = Var("jetArea()", float, doc="jet catchment area, for JECs",precision=10),
        # energy fractions
        chHEF = Var("chargedHadronEnergyFraction()", float, doc="charged Hadron Energy Fraction", precision= 6),
        neHEF = Var("neutralHadronEnergyFraction()", float, doc="neutral Hadron Energy Fraction", precision= 6),
        chEmEF = Var("chargedEmEnergyFraction()", float, doc="charged Electromagnetic Energy Fraction", precision= 6),
        neEmEF = Var("neutralEmEnergyFraction()", float, doc="neutral Electromagnetic Energy Fraction", precision= 6),
        muEF = Var("muonEnergyFraction()", float, doc="muon Energy Fraction", precision= 6),
        hfHEF = Var("HFHadronEnergyFraction()",float,doc="hadronic Energy Fraction in HF",precision= 6),
        hfEmEF = Var("HFEMEnergyFraction()",float,doc="electromagnetic Energy Fraction in HF",precision= 6),
        # multiplicities
        nCh = Var("chargedHadronMultiplicity()", int, doc="number of charged hadrons in the jet"),
        nNh = Var("neutralHadronMultiplicity()", int, doc="number of neutral hadrons in the jet"),
        nMuons = Var("muonMultiplicity()", int, doc="number of muons in the jet"),
        nElectrons = Var("electronMultiplicity()", int, doc="number of electrons in the jet"),
        nPhotons = Var("photonMultiplicity()", int, doc="number of photons in the jet"),
        nConstituents = Var("numberOfDaughters()", "uint8", doc="number of particles in the jet")
    ),
    externalVariables = cms.PSet(
        # jet tagging probabilities
        particleNet_prob_b = ExtVar(cms.InputTag("scoutingPFJetReclusterParticleNetJetTags:probb"), float, doc="ParticleNet probability of b", precision=10),
        particleNet_prob_bb = ExtVar(cms.InputTag("scoutingPFJetReclusterParticleNetJetTags:probbb"), float, doc="ParticleNet probability of bb", precision=10),
        particleNet_prob_c = ExtVar(cms.InputTag("scoutingPFJetReclusterParticleNetJetTags:probc"), float, doc="ParticleNet probability of c", precision=10),
        particleNet_prob_cc = ExtVar(cms.InputTag("scoutingPFJetReclusterParticleNetJetTags:probcc"), float, doc="ParticleNet probability of cc", precision=10),
        particleNet_prob_uds = ExtVar(cms.InputTag("scoutingPFJetReclusterParticleNetJetTags:probuds"), float, doc="particlenet probability of uds", precision=10),
        particleNet_prob_g = ExtVar(cms.InputTag("scoutingPFJetReclusterParticleNetJetTags:probg"), float, doc="ParticleNet probability of g", precision=10),
        particleNet_prob_undef = ExtVar(cms.InputTag("scoutingPFJetReclusterParticleNetJetTags:probundef"), float, doc="ParticleNet probability of undef", precision=10),
    ),
)

# AK4 gen jet matching (only for MC)

scoutingPFJetReclusterMatchGen = cms.EDProducer("RecoJetToGenJetDeltaRValueMapProducer",
    src = cms.InputTag("scoutingPFJetRecluster"),
    matched = cms.InputTag("slimmedGenJets"),
    distMax = cms.double(0.4),
    value = cms.string("index"),
)

scoutingPFJetReclusterMatchGenExtensionTable = cms.EDProducer("SimplePFJetFlatTableProducer",
    src = cms.InputTag("scoutingPFJetRecluster"),
    name = cms.string("ScoutingPFJetRecluster"),
    cut = cms.string(""),
    singleton = cms.bool(False),
    extension = cms.bool(True),
    variables = cms.PSet(),
    externalVariables = cms.PSet(
        genJetIdx = ExtVar(cms.InputTag("scoutingPFJetReclusterMatchGen"), int, doc="gen jet idx"),
    ),
)

#########################
# AK8 PFJet Reclustered #
#########################

# AK8 jet clustering

scoutingFatPFJetRecluster = ak4PFJets.clone(
    src = ("scoutingPFCandidate"),
    rParam   = 0.8,
    jetPtMin = 170.0,
)

# AK8 jet tagging

scoutingFatPFJetReclusterParticleNetJetTagInfos = cms.EDProducer("DeepBoostedJetTagInfoProducer",
    jet_radius = cms.double(0.8),
    min_jet_pt = cms.double(50),
    max_jet_eta = cms.double(2.5),
    min_pt_for_track_properties = cms.double(0.95),
    min_pt_for_pfcandidates = cms.double(0.1),
    use_puppiP4 = cms.bool(False),
    include_neutrals = cms.bool(True),
    sort_by_sip2dsig = cms.bool(False),
    min_puppi_wgt = cms.double(-1.0),
    flip_ip_sign = cms.bool(False),
    sip3dSigMax = cms.double(-1.0),
    use_hlt_features = cms.bool(False),
    pf_candidates = cms.InputTag("scoutingPFCandidate"),
    jets = cms.InputTag("scoutingFatPFJetRecluster"),
    puppi_value_map = cms.InputTag(""),
    use_scouting_features = cms.bool(True),
    normchi2_value_map = cms.InputTag("scoutingPFCandidate", "normchi2"),
    dz_value_map = cms.InputTag("scoutingPFCandidate", "dz"),
    dxy_value_map = cms.InputTag("scoutingPFCandidate", "dxy"),
    dzsig_value_map = cms.InputTag("scoutingPFCandidate", "dzsig"),
    dxysig_value_map = cms.InputTag("scoutingPFCandidate", "dxysig"),
    lostInnerHits_value_map = cms.InputTag("scoutingPFCandidate", "lostInnerHits"),
    quality_value_map = cms.InputTag("scoutingPFCandidate", "quality"),
    trkPt_value_map = cms.InputTag("scoutingPFCandidate", "trkPt"),
    trkEta_value_map = cms.InputTag("scoutingPFCandidate", "trkEta"),
    trkPhi_value_map = cms.InputTag("scoutingPFCandidate", "trkPhi"),
)

from RecoBTag.ONNXRuntime.boostedJetONNXJetTagsProducer_cfi import boostedJetONNXJetTagsProducer
scoutingFatPFJetReclusterParticleNetJetTags = cms.EDProducer("BoostedJetONNXJetTagsProducer",
    jets = cms.InputTag("scoutingFatPFJetRecluster"),
    produceValueMap = cms.untracked.bool(True),
    src = cms.InputTag("scoutingFatPFJetReclusterParticleNetJetTagInfos"),
    preprocess_json = cms.string("RecoBTag/Combined/data/Run3Scouting/ParticleNetAK8/General/V00/preprocess.json"),
    model_path = cms.FileInPath("RecoBTag/Combined/data/Run3Scouting/ParticleNetAK8/General/V00/particle-net.onnx"),
    flav_names = cms.vstring(["probQCDall", "probHbb","probHcc", "probHqq"]),
    debugMode = cms.untracked.bool(False),
)

scoutingFatPFJetReclusterGlobalParticleTransformerJetTagInfos = cms.EDProducer("DeepBoostedJetTagInfoProducer",
    jet_radius = cms.double(0.8),
    min_jet_pt = cms.double(50),
    max_jet_eta = cms.double(2.5),
    min_pt_for_track_properties = cms.double(0.95),
    min_pt_for_pfcandidates = cms.double(0.1),
    use_puppiP4 = cms.bool(False),
    include_neutrals = cms.bool(True),
    sort_by_sip2dsig = cms.bool(False),
    min_puppi_wgt = cms.double(-1.0),
    flip_ip_sign = cms.bool(False),
    sip3dSigMax = cms.double(-1.0),
    use_hlt_features = cms.bool(False),
    pf_candidates = cms.InputTag("scoutingPFCandidate"),
    jets = cms.InputTag("scoutingFatPFJetRecluster"),
    puppi_value_map = cms.InputTag(""),
    use_scouting_features = cms.bool(True),
    normchi2_value_map = cms.InputTag("scoutingPFCandidate", "normchi2"),
    dz_value_map = cms.InputTag("scoutingPFCandidate", "dz"),
    dxy_value_map = cms.InputTag("scoutingPFCandidate", "dxy"),
    dzsig_value_map = cms.InputTag("scoutingPFCandidate", "dzsig"),
    dxysig_value_map = cms.InputTag("scoutingPFCandidate", "dxysig"),
    lostInnerHits_value_map = cms.InputTag("scoutingPFCandidate", "lostInnerHits"),
    quality_value_map = cms.InputTag("scoutingPFCandidate", "quality"),
    trkPt_value_map = cms.InputTag("scoutingPFCandidate", "trkPt"),
    trkEta_value_map = cms.InputTag("scoutingPFCandidate", "trkEta"),
    trkPhi_value_map = cms.InputTag("scoutingPFCandidate", "trkPhi"),
)

scoutingFatPFJetReclusterGlobalParticleTransformerJetTags = cms.EDProducer("BoostedJetONNXJetTagsProducer",
    jets = cms.InputTag("scoutingFatPFJetRecluster"),
    produceValueMap = cms.untracked.bool(True),
    src = cms.InputTag("scoutingFatPFJetReclusterGlobalParticleTransformerJetTagInfos"),
    preprocess_json = cms.string("RecoBTag/Combined/data/Run3Scouting/GlobalParticleTransformerAK8/General/V00/preprocess.json"),
    model_path = cms.FileInPath("RecoBTag/Combined/data/Run3Scouting/GlobalParticleTransformerAK8/General/V00/global-part_2024.onnx"),
    flav_names = cms.vstring([
             "probQCD", "probXbb", "probXcc", "probXss", "probXqq", "probXbs", "probXgg", "probXee", "probXmm", "probXtauhtaue", "probXtauhtaum", "probXtauhtauh", "probXbc", "probXcs", "probXud", "massCorrGeneric", "massCorrGenericX2p", "massCorrGenericW2p", "massCorrResonance"
     ]),
    debugMode = cms.untracked.bool(False),
)

# AK8 jet softdrop mass

scoutingFatPFJetReclusterSoftDrop = ak4PFJets.clone(
    src = ("scoutingPFCandidate"),
    rParam   = 0.8,
    jetPtMin = 170.0,
    useSoftDrop = cms.bool(True),
    zcut = cms.double(0.1),
    beta = cms.double(0.0),
    R0   = cms.double(0.8),
    useExplicitGhosts = cms.bool(True),
    writeCompound = cms.bool(True),
    jetCollInstanceName=cms.string("SubJets"),
)

scoutingFatPFJetReclusterSoftDropMass = cms.EDProducer("RecoJetDeltaRValueMapProducer",
    src = cms.InputTag("scoutingFatPFJetRecluster"),
    matched = cms.InputTag("scoutingFatPFJetReclusterSoftDrop"),
    distMax = cms.double(0.8),
    value = cms.string("mass")
)

# AK8 jet regressed mass

scoutingFatPFJetReclusterParticleNetMassRegressionJetTags = cms.EDProducer("BoostedJetONNXJetTagsProducer",
    jets = cms.InputTag("scoutingFatPFJetRecluster"),
    produceValueMap = cms.untracked.bool(True),
    src = cms.InputTag("scoutingFatPFJetReclusterParticleNetJetTagInfos"),
    preprocess_json = cms.string("RecoBTag/Combined/data/Run3Scouting/ParticleNetAK8/MassRegression/V00/preprocess.json"),
    model_path = cms.FileInPath("RecoBTag/Combined/data/Run3Scouting/ParticleNetAK8/MassRegression/V00/particle-net.onnx"),
    flav_names = cms.vstring(["mass"]),
    debugMode = cms.untracked.bool(False),
)

# AK8 jet substructure variables 

from RecoJets.JetProducers.ECF_cff import ecfNbeta1
scoutingFatPFJetReclusterEcfNbeta1 = ecfNbeta1.clone(src = cms.InputTag("scoutingFatPFJetRecluster"), srcWeights="")

from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness
scoutingFatPFJetReclusterNjettiness = Njettiness.clone(src = cms.InputTag("scoutingFatPFJetRecluster"), srcWeights="")

# output AK8 jet to nanoaod::flattable

scoutingFatPFJetReclusterTable = cms.EDProducer("SimplePFJetFlatTableProducer",
    src = cms.InputTag("scoutingFatPFJetRecluster"),
    name = cms.string("ScoutingFatPFJetRecluster"),
    cut = cms.string(""),
    doc = cms.string("ak8 jet from re-clustering scouting PF Candidates"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = cms.PSet(
        P4Vars,
        area = Var("jetArea()", float, doc="jet catchment area, for JECs",precision=10),
        # energy fractions
        chHEF = Var("chargedHadronEnergyFraction()", float, doc="charged Hadron Energy Fraction", precision= 6),
        neHEF = Var("neutralHadronEnergyFraction()", float, doc="neutral Hadron Energy Fraction", precision= 6),
        chEmEF = Var("chargedEmEnergyFraction()", float, doc="charged Electromagnetic Energy Fraction", precision= 6),
        neEmEF = Var("neutralEmEnergyFraction()", float, doc="neutral Electromagnetic Energy Fraction", precision= 6),
        muEF = Var("muonEnergyFraction()", float, doc="muon Energy Fraction", precision= 6),
        hfHEF = Var("HFHadronEnergyFraction()",float,doc="hadronic Energy Fraction in HF",precision= 6),
        hfEmEF = Var("HFEMEnergyFraction()",float,doc="electromagnetic Energy Fraction in HF",precision= 6),
        # multiplicities
        nCh = Var("chargedHadronMultiplicity()", int, doc="number of charged hadrons in the jet"),
        nNh = Var("neutralHadronMultiplicity()", int, doc="number of neutral hadrons in the jet"),
        nMuons = Var("muonMultiplicity()", int, doc="number of muons in the jet"),
        nElectrons = Var("electronMultiplicity()", int, doc="number of electrons in the jet"),
        nPhotons = Var("photonMultiplicity()", int, doc="number of photons in the jet"),
        nConstituents = Var("numberOfDaughters()", "uint8", doc="number of particles in the jet")
    ),
    externalVariables = cms.PSet(
        # jet tagging probabilities
        particleNet_prob_QCD = ExtVar(cms.InputTag("scoutingFatPFJetReclusterParticleNetJetTags:probQCDall"), float, doc="ParticleNet probability of QCD", precision=10),
        particleNet_prob_Hbb = ExtVar(cms.InputTag("scoutingFatPFJetReclusterParticleNetJetTags:probHbb"), float, doc="ParticleNet probability of Hbb", precision=10),
        particleNet_prob_Hcc = ExtVar(cms.InputTag("scoutingFatPFJetReclusterParticleNetJetTags:probHcc"), float, doc="ParticleNet probability of Hcc", precision=10),
        particleNet_prob_Hqq = ExtVar(cms.InputTag("scoutingFatPFJetReclusterParticleNetJetTags:probHqq"), float, doc="ParticleNet probability of Hqq", precision=10),
        scoutGlobalParT_prob_QCD = ExtVar(cms.InputTag("scoutingFatPFJetReclusterGlobalParticleTransformerJetTags:probQCD"), float, doc="Mass-decorrelated Scouting GlobalParT QCD score", precision=10),
        scoutGlobalParT_prob_Xbb = ExtVar(cms.InputTag("scoutingFatPFJetReclusterGlobalParticleTransformerJetTags:probXbb"), float, doc="Mass-decorrelated Scouting GlobalParT X->bb score", precision=10),
        scoutGlobalParT_prob_Xcc = ExtVar(cms.InputTag("scoutingFatPFJetReclusterGlobalParticleTransformerJetTags:probXcc"), float, doc="Mass-decorrelated Scouting GlobalParT X->cc score", precision=10),
        scoutGlobalParT_prob_Xss = ExtVar(cms.InputTag("scoutingFatPFJetReclusterGlobalParticleTransformerJetTags:probXss"), float, doc="Mass-decorrelated Scouting GlobalParT X->ss score", precision=10),
        scoutGlobalParT_prob_Xqq = ExtVar(cms.InputTag("scoutingFatPFJetReclusterGlobalParticleTransformerJetTags:probXqq"), float, doc="Mass-decorrelated Scouting GlobalParT X->qq score", precision=10),
        scoutGlobalParT_prob_Xbc = ExtVar(cms.InputTag("scoutingFatPFJetReclusterGlobalParticleTransformerJetTags:probXbc"), float, doc="Mass-decorrelated Scouting GlobalParT X->bc score", precision=10),
        scoutGlobalParT_prob_Xbs = ExtVar(cms.InputTag("scoutingFatPFJetReclusterGlobalParticleTransformerJetTags:probXbs"), float, doc="Mass-decorrelated Scouting GlobalParT X->bs score", precision=10),
        scoutGlobalParT_prob_Xcs = ExtVar(cms.InputTag("scoutingFatPFJetReclusterGlobalParticleTransformerJetTags:probXcs"), float, doc="Mass-decorrelated Scouting GlobalParT X->cs score", precision=10),
        scoutGlobalParT_prob_Xud = ExtVar(cms.InputTag("scoutingFatPFJetReclusterGlobalParticleTransformerJetTags:probXud"), float, doc="Mass-decorrelated Scouting GlobalParT X->ud score", precision=10),
        scoutGlobalParT_prob_Xgg = ExtVar(cms.InputTag("scoutingFatPFJetReclusterGlobalParticleTransformerJetTags:probXgg"), float, doc="Mass-decorrelated Scouting GlobalParT X->gg score", precision=10),
        scoutGlobalParT_prob_Xtauhtaue = ExtVar(cms.InputTag("scoutingFatPFJetReclusterGlobalParticleTransformerJetTags:probXtauhtaue"), float, doc="Mass-decorrelated Scouting GlobalParT X->tauhtaue score", precision=10),
        scoutGlobalParT_prob_Xtauhtaum = ExtVar(cms.InputTag("scoutingFatPFJetReclusterGlobalParticleTransformerJetTags:probXtauhtaum"), float, doc="Mass-decorrelated Scouting GlobalParT X->tauhtaum score", precision=10),
        scoutGlobalParT_prob_Xtauhtauh = ExtVar(cms.InputTag("scoutingFatPFJetReclusterGlobalParticleTransformerJetTags:probXtauhtauh"), float, doc="Mass-decorrelated Scouting GlobalParT X->tauhtauh score", precision=10),
        # softdrop mass
        msoftdrop = ExtVar(cms.InputTag("scoutingFatPFJetReclusterSoftDropMass"), float, doc="Softdrop mass", precision=10),
        # regressed mass
        particleNet_mass = ExtVar(cms.InputTag("scoutingFatPFJetReclusterParticleNetMassRegressionJetTags:mass"), float, doc="ParticleNet regressed mass", precision=10),
        scoutGlobalParT_massCorrGeneric = ExtVar(cms.InputTag("scoutingFatPFJetReclusterGlobalParticleTransformerJetTags:massCorrGeneric"), float, doc="Mass-decorrelated Scouting GlobalParT mass regression corrector with respect to the original jet mass, optimised for generic jet cases. Use (massCorrGeneric * mass) to get the regressed mass", precision=10),
        scoutGlobalParT_massCorrGenericX2p = ExtVar(cms.InputTag("scoutingFatPFJetReclusterGlobalParticleTransformerJetTags:massCorrGenericX2p"), float, doc="Mass-decorrelated Scouting GlobalParT mass regression corrector with respect to the original jet mass, optimised for generic X2p jet cases. Use (massCorrGenericX2p * mass) to get the regressed mass", precision=10),
        scoutGlobalParT_massCorrGenericW2p = ExtVar(cms.InputTag("scoutingFatPFJetReclusterGlobalParticleTransformerJetTags:massCorrGenericW2p"), float, doc="Mass-decorrelated Scouting GlobalParT mass regression corrector with respect to the original jet mass, optimised for generic W jet cases. Use (massCorrGenericW2p * mass) to get the regressed mass", precision=10),
        scoutGlobalParT_massCorrResonance = ExtVar(cms.InputTag("scoutingFatPFJetReclusterGlobalParticleTransformerJetTags:massCorrResonance"), float, doc="Scouting GlobalParT mass regression corrector with respect to the original jet mass, optimised for resonance jets. Use (massCorrResonance * mass) to get the regressed mass", precision=10),
        # substructure variables    
        n2b1 = ExtVar(cms.InputTag("scoutingFatPFJetReclusterEcfNbeta1:ecfN2"), float, doc="N2 with beta=1", precision=10),
        n3b1 = ExtVar(cms.InputTag("scoutingFatPFJetReclusterEcfNbeta1:ecfN3"), float, doc="N3 with beta=1", precision=10),
        tau1 = ExtVar(cms.InputTag("scoutingFatPFJetReclusterNjettiness:tau1"), float, doc="Nsubjettiness (1 axis)", precision=10),
        tau2 = ExtVar(cms.InputTag("scoutingFatPFJetReclusterNjettiness:tau2"), float, doc="Nsubjettiness (2 axis)", precision=10),
        tau3 = ExtVar(cms.InputTag("scoutingFatPFJetReclusterNjettiness:tau3"), float, doc="Nsubjettiness (3 axis)", precision=10),
        tau4 = ExtVar(cms.InputTag("scoutingFatPFJetReclusterNjettiness:tau4"), float, doc="Nsubjettiness (4 axis)", precision=10),
    ),
)

# AK8 gen jet matching (only for MC)

scoutingFatPFJetReclusterMatchGen = cms.EDProducer("RecoJetToGenJetDeltaRValueMapProducer",
    src = cms.InputTag("scoutingFatPFJetRecluster"),
    matched = cms.InputTag("slimmedGenJetsAK8"),
    distMax = cms.double(0.8),
    value = cms.string("index"),
)

scoutingFatPFJetReclusterMatchGenExtensionTable = cms.EDProducer("SimplePFJetFlatTableProducer",
    src = cms.InputTag("scoutingFatPFJetRecluster"),
    name = cms.string("ScoutingFatPFJetRecluster"),
    cut = cms.string(""),
    singleton = cms.bool(False),
    extension = cms.bool(True),
    variables = cms.PSet(),
    externalVariables = cms.PSet(
        genJetAK8Idx = ExtVar(cms.InputTag("scoutingFatPFJetReclusterMatchGen"), int, doc="gen jet idx"),
    ),
)
