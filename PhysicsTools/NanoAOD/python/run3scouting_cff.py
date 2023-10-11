import FWCore.ParameterSet.Config as cms
from  PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.simpleCandidateFlatTableProducer_cfi import simpleCandidateFlatTableProducer

################
# Scouting photons, electrons, muons, tracks, primary vertices, displaced vertices, rho and MET

photonScoutingTable = cms.EDProducer("SimpleRun3ScoutingPhotonFlatTableProducer",
     src = cms.InputTag("hltScoutingEgammaPacker"),
     cut = cms.string(""),
     name = cms.string("ScoutingPhoton"),
     doc  = cms.string("Photon scouting information"),
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
         r9 = Var('r9', 'float', precision=10, doc='Photon SC r9 as defined in https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideEgammaShowerShape'),
         sMin = Var('sMin', 'float', precision=10, doc='minor moment of the SC shower shape'),
         sMaj = Var('sMaj', 'float', precision=10, doc='major moment of the SC shower shape'),
         seedId = Var('seedId', 'int', doc='ECAL ID of the SC seed'),
     )
)

electronScoutingTable = cms.EDProducer("SimpleRun3ScoutingElectronFlatTableProducer",
     src = cms.InputTag("hltScoutingEgammaPacker"),
     cut = cms.string(""),
     name = cms.string("ScoutingElectron"),
     doc  = cms.string("Electron scouting information"),
     singleton = cms.bool(False),
     extension = cms.bool(False),
     variables = cms.PSet(
         pt = Var('pt', 'float', precision=10, doc='super-cluster (SC) pt'),
         eta = Var('eta', 'float', precision=10, doc='SC eta'),
         phi = Var('phi', 'float', precision=10, doc='SC phi'),
         m = Var('m', 'float', precision=10, doc='SC mass'),
         dEtaIn = Var('dEtaIn', 'float', precision=10, doc='#Delta#eta(SC seed, track pixel seed)'),
         dPhiIn = Var('dPhiIn', 'float', precision=10, doc='#Delta#phi(SC seed, track pixel seed)'),
         sigmaIetaIeta = Var('sigmaIetaIeta', 'float', precision=10, doc='sigmaIetaIeta of the SC, calculated with full 5x5 region, noise cleaned'),
         hOverE = Var('hOverE', 'float', precision=10, doc='Energy in HCAL / Energy in ECAL'),
         ooEMOop = Var('ooEMOop', 'float', precision=10, doc='1/E(SC) - 1/p(track momentum)'),
         missingHits = Var('missingHits', 'int', doc='missing hits in the tracker'),
         ecalIso = Var('ecalIso', 'float', precision=10, doc='Isolation of SC in the ECAL'),
         hcalIso = Var('hcalIso', 'float', precision=10, doc='Isolation of SC in the HCAL'),
         trackIso = Var('trackIso', 'float', precision=10, doc='Isolation of electron track in the tracker'),
         r9 = Var('r9', 'float', precision=10, doc='ELectron SC r9 as defined in https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideEgammaShowerShape'),
         sMin = Var('sMin', 'float', precision=10, doc='minor moment of the SC shower shape'),
         sMaj = Var('sMaj', 'float', precision=10, doc='major moment of the SC shower shape'),
         seedId = Var('seedId', 'int', doc='ECAL ID of the SC seed'),
     )
)

muonScoutingTable = cms.EDProducer("SimpleRun3ScoutingMuonFlatTableProducer",
     src = cms.InputTag("hltScoutingMuonPacker"),
     cut = cms.string(""),
     name = cms.string("ScoutingMuon"),
     doc  = cms.string("Muon scouting information"),
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
     )
)

trackScoutingTable = cms.EDProducer("SimpleRun3ScoutingTrackFlatTableProducer",
     src = cms.InputTag("hltScoutingTrackPacker"),
     cut = cms.string(""),
     name = cms.string("ScoutingTrack"),
     doc  = cms.string("Track scouting information"),
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
         _lambda = Var('tk_lambda', 'float', precision=10, doc='lambda'),
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
     )
)

primaryvertexScoutingTable = cms.EDProducer("SimpleRun3ScoutingVertexFlatTableProducer",
     src = cms.InputTag("hltScoutingPrimaryVertexPacker", "primaryVtx"),
     cut = cms.string(""),
     name = cms.string("ScoutingPrimaryVertex"),
     doc  = cms.string("PrimaryVertex scouting information"),
     singleton = cms.bool(False),
     extension = cms.bool(False),
     variables = cms.PSet(
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
)

displacedvertexScoutingTable = cms.EDProducer("SimpleRun3ScoutingVertexFlatTableProducer",
     src = cms.InputTag("hltScoutingMuonPacker","displacedVtx"),
     cut = cms.string(""),
     name = cms.string("ScoutingDisplacedVertex"),
     doc  = cms.string("DisplacedVertex scouting information"),
     singleton = cms.bool(False),
     extension = cms.bool(False),
     variables = cms.PSet(
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
)

rhoScoutingTable = cms.EDProducer("GlobalVariablesTableProducer",
    name = cms.string(""),
    variables = cms.PSet(
        ScoutingRho = ExtVar( cms.InputTag("hltScoutingPFPacker", "rho"), "double", doc = "rho from all scouting PF Candidates, used e.g. for JECs" ),
    )
)

metScoutingTable = cms.EDProducer("GlobalVariablesTableProducer",
    name = cms.string("ScoutingMET"),
    variables = cms.PSet(
        pt = ExtVar( cms.InputTag("hltScoutingPFPacker", "pfMetPt"), "double", doc = "scouting MET pt"),
        phi = ExtVar( cms.InputTag("hltScoutingPFPacker", "pfMetPhi"), "double", doc = "scouting MET phi"),
    )
)

################
# Scouting particles

scoutingPFCands = cms.EDProducer(
     "Run3ScoutingParticleToRecoPFCandidateProducer",
     scoutingparticle=cms.InputTag("hltScoutingPFPacker"),
)

particleScoutingTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("scoutingPFCands"),
    name = cms.string("ScoutingParticle"),
    cut = cms.string(""),
    doc = cms.string("ScoutingParticle"),
    singleton = cms.bool(False),
    extension = cms.bool(False), # this is the main table
    externalVariables = cms.PSet(
       vertexIndex = ExtVar(cms.InputTag("scoutingPFCands", "vertexIndex"), int, doc="vertex index"),
       trkNormchi2 = ExtVar(cms.InputTag("scoutingPFCands", "normchi2"), float, doc="normalized chi squared of best track", precision=6),
       trkDz = ExtVar(cms.InputTag("scoutingPFCands", "dz"), float, doc="dz of best track", precision=6),
       trkDxy = ExtVar(cms.InputTag("scoutingPFCands", "dxy"), float, doc="dxy of best track", precision=6),
       trkDzsig = ExtVar(cms.InputTag("scoutingPFCands", "dzsig"), float, doc="dzsig of best track", precision=6),
       trkDxysig = ExtVar(cms.InputTag("scoutingPFCands", "dxysig"), float, doc="dxysig of best track", precision=6),
       trkLostInnerHits = ExtVar(cms.InputTag("scoutingPFCands", "lostInnerHits"), int, doc="lostInnerHits of best track"),
       trkQuality = ExtVar(cms.InputTag("scoutingPFCands", "quality"), int, doc="quality of best track"),
       trkPt = ExtVar(cms.InputTag("scoutingPFCands", "trkPt"), float, doc="pt of best track", precision=6),
       trkEta = ExtVar(cms.InputTag("scoutingPFCands", "trkEta"), float, doc="eta of best track", precision=6),
       trkPhi = ExtVar(cms.InputTag("scoutingPFCands", "trkPhi"), float, doc="phi of best track", precision=6),
    ),
    variables = cms.PSet(
       CandVars,
    ),
  )

################
# Scouting AK4 jets

from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJets
ak4ScoutingJets = ak4PFJets.clone(
     src = ("scoutingPFCands"),
     jetPtMin = 20,
)

ak4ScoutingJetParticleNetJetTagInfos = cms.EDProducer("DeepBoostedJetTagInfoProducer",
      jet_radius = cms.double( 0.4 ),
      min_jet_pt = cms.double( 5.0 ),
      max_jet_eta = cms.double( 2.5 ),
      min_pt_for_track_properties = cms.double( 0.95 ),
      min_pt_for_pfcandidates = cms.double( 0.1 ),
      use_puppiP4 = cms.bool( False ),
      include_neutrals = cms.bool( True ),
      sort_by_sip2dsig = cms.bool( False ),
      min_puppi_wgt = cms.double( -1.0 ),
      flip_ip_sign = cms.bool( False ),
      sip3dSigMax = cms.double( -1.0 ),
      use_hlt_features = cms.bool( False ),
      pf_candidates = cms.InputTag( "scoutingPFCands" ),
      jets = cms.InputTag( "ak4ScoutingJets" ),
      puppi_value_map = cms.InputTag( "" ),
      use_scouting_features = cms.bool( True ),
      normchi2_value_map = cms.InputTag("scoutingPFCands", "normchi2"),
      dz_value_map = cms.InputTag("scoutingPFCands", "dz"),
      dxy_value_map = cms.InputTag("scoutingPFCands", "dxy"),
      dzsig_value_map = cms.InputTag("scoutingPFCands", "dzsig"),
      dxysig_value_map = cms.InputTag("scoutingPFCands", "dxysig"),
      lostInnerHits_value_map = cms.InputTag("scoutingPFCands", "lostInnerHits"),
      quality_value_map = cms.InputTag("scoutingPFCands", "quality"),
      trkPt_value_map = cms.InputTag("scoutingPFCands", "trkPt"),
      trkEta_value_map = cms.InputTag("scoutingPFCands", "trkEta"),
      trkPhi_value_map = cms.InputTag("scoutingPFCands", "trkPhi"),
)

from RecoBTag.ONNXRuntime.boostedJetONNXJetTagsProducer_cfi import boostedJetONNXJetTagsProducer

ak4ScoutingJetParticleNetJetTags = cms.EDProducer("BoostedJetONNXJetTagsProducer",
      jets = cms.InputTag("ak4ScoutingJets"),
      produceValueMap = cms.untracked.bool(True),
      src = cms.InputTag("ak4ScoutingJetParticleNetJetTagInfos"),
      preprocess_json = cms.string("RecoBTag/Combined/data/Run3Scouting/ParticleNetAK4/V00/preprocess.json"),
      model_path = cms.FileInPath("RecoBTag/Combined/data/Run3Scouting/ParticleNetAK4/V00/particle-net.onnx"),
      flav_names = cms.vstring(["probb", "probbb","probc", "probcc", "probuds", "probg", "probundef"]),
      debugMode = cms.untracked.bool(False),
)

ak4ScoutingJetTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
      src = cms.InputTag("ak4ScoutingJets"),
      name = cms.string("ScoutingJet"),
      cut = cms.string(""),
      doc = cms.string("ScoutingJet"),
      singleton = cms.bool(False),
      extension = cms.bool(False), # this is the main table
      externalVariables = cms.PSet(
         particleNet_prob_b = ExtVar(cms.InputTag('ak4ScoutingJetParticleNetJetTags:probb'), float, doc="ParticleNet probability of b", precision=10),
         particleNet_prob_bb = ExtVar(cms.InputTag('ak4ScoutingJetParticleNetJetTags:probbb'), float, doc="ParticleNet probability of bb", precision=10),
         particleNet_prob_c = ExtVar(cms.InputTag('ak4ScoutingJetParticleNetJetTags:probc'), float, doc="ParticleNet probability of c", precision=10),
         particleNet_prob_cc = ExtVar(cms.InputTag('ak4ScoutingJetParticleNetJetTags:probcc'), float, doc="ParticleNet probability of cc", precision=10),
         particlenet_prob_uds = ExtVar(cms.InputTag('ak4ScoutingJetParticleNetJetTags:probuds'), float, doc="particlenet probability of uds", precision=10),
         particleNet_prob_g = ExtVar(cms.InputTag('ak4ScoutingJetParticleNetJetTags:probg'), float, doc="ParticleNet probability of g", precision=10),
         particleNet_prob_undef = ExtVar(cms.InputTag('ak4ScoutingJetParticleNetJetTags:probundef'), float, doc="ParticleNet probability of undef", precision=10),
      ),
      variables = cms.PSet(
         P4Vars,
         area = Var("jetArea()", float, doc="jet catchment area, for JECs",precision=10),
         chHEF = Var("chargedHadronEnergy()/(chargedHadronEnergy()+neutralHadronEnergy()+photonEnergy()+electronEnergy()+muonEnergy())", float, doc="charged Hadron Energy Fraction", precision= 6),
         neHEF = Var("neutralHadronEnergy()/(chargedHadronEnergy()+neutralHadronEnergy()+photonEnergy()+electronEnergy()+muonEnergy())", float, doc="neutral Hadron Energy Fraction", precision= 6),
         chEmEF = Var("(electronEnergy()+muonEnergy())/(chargedHadronEnergy()+neutralHadronEnergy()+photonEnergy()+electronEnergy()+muonEnergy())", float, doc="charged Electromagnetic Energy Fraction", precision= 6),
         neEmEF = Var("(photonEnergy())/(chargedHadronEnergy()+neutralHadronEnergy()+photonEnergy()+electronEnergy()+muonEnergy())", float, doc="neutral Electromagnetic Energy Fraction", precision= 6),
         muEF = Var("(muonEnergy())/(chargedHadronEnergy()+neutralHadronEnergy()+photonEnergy()+electronEnergy()+muonEnergy())", float, doc="muon Energy Fraction", precision= 6),
         nCh = Var("chargedHadronMultiplicity()", int, doc="number of charged hadrons in the jet"),
         nNh = Var("neutralHadronMultiplicity()", int, doc="number of neutral hadrons in the jet"),
         nMuons = Var("muonMultiplicity()", int, doc="number of muons in the jet"),
         nElectrons = Var("electronMultiplicity()", int, doc="number of electrons in the jet"),
         nPhotons = Var("photonMultiplicity()", int, doc="number of photons in the jet"),
         nConstituents = Var("numberOfDaughters()", "uint8", doc="number of particles in the jet")
      ),
)

ak4ScoutingJetMatchGen = cms.EDProducer("RecoJetToGenJetDeltaRValueMapProducer",
      src = cms.InputTag("ak4ScoutingJets"),
      matched = cms.InputTag("slimmedGenJets"),
      distMax = cms.double(0.4),
      value = cms.string("index"),
  )

ak4ScoutingJetExtTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
      src = cms.InputTag("ak4ScoutingJets"),
      name = cms.string("ScoutingJet"),
      cut = cms.string(""),
      singleton = cms.bool(False),
      extension = cms.bool(True),
      externalVariables = cms.PSet(
         genJetIdx = ExtVar(cms.InputTag("ak4ScoutingJetMatchGen"), int, doc="gen jet idx"),
      ),
      variables = cms.PSet(),
  )

################
# Scouting AK8 jets

ak8ScoutingJets = ak4PFJets.clone(
     src = ("scoutingPFCands"),
     rParam   = 0.8,
     jetPtMin = 170.0,
)

ak8ScoutingJetsSoftDrop = ak4PFJets.clone(
     src = ("scoutingPFCands"),
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

ak8ScoutingJetsSoftDropMass = cms.EDProducer("RecoJetDeltaRValueMapProducer",
     src = cms.InputTag("ak8ScoutingJets"),
     matched = cms.InputTag("ak8ScoutingJetsSoftDrop"),
     distMax = cms.double(0.8),
     value = cms.string('mass')
  )

from RecoJets.JetProducers.ECF_cff import ecfNbeta1
ak8ScoutingJetEcfNbeta1 = ecfNbeta1.clone(src = cms.InputTag("ak8ScoutingJets"), srcWeights="")

from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness
ak8ScoutingJetNjettiness = Njettiness.clone(src = cms.InputTag("ak8ScoutingJets"), srcWeights="")

ak8ScoutingJetParticleNetJetTagInfos = cms.EDProducer("DeepBoostedJetTagInfoProducer",
      jet_radius = cms.double( 0.8 ),
      min_jet_pt = cms.double( 50 ),
      max_jet_eta = cms.double( 2.5 ),
      min_pt_for_track_properties = cms.double( 0.95 ),
      min_pt_for_pfcandidates = cms.double( 0.1 ),
      use_puppiP4 = cms.bool( False ),
      include_neutrals = cms.bool( True ),
      sort_by_sip2dsig = cms.bool( False ),
      min_puppi_wgt = cms.double( -1.0 ),
      flip_ip_sign = cms.bool( False ),
      sip3dSigMax = cms.double( -1.0 ),
      use_hlt_features = cms.bool( False ),
      pf_candidates = cms.InputTag( "scoutingPFCands" ),
      jets = cms.InputTag( "ak8ScoutingJets" ),
      puppi_value_map = cms.InputTag( "" ),
      use_scouting_features = cms.bool( True ),
      normchi2_value_map = cms.InputTag("scoutingPFCands", "normchi2"),
      dz_value_map = cms.InputTag("scoutingPFCands", "dz"),
      dxy_value_map = cms.InputTag("scoutingPFCands", "dxy"),
      dzsig_value_map = cms.InputTag("scoutingPFCands", "dzsig"),
      dxysig_value_map = cms.InputTag("scoutingPFCands", "dxysig"),
      lostInnerHits_value_map = cms.InputTag("scoutingPFCands", "lostInnerHits"),
      quality_value_map = cms.InputTag("scoutingPFCands", "quality"),
      trkPt_value_map = cms.InputTag("scoutingPFCands", "trkPt"),
      trkEta_value_map = cms.InputTag("scoutingPFCands", "trkEta"),
      trkPhi_value_map = cms.InputTag("scoutingPFCands", "trkPhi"),
  )

from RecoBTag.ONNXRuntime.boostedJetONNXJetTagsProducer_cfi import boostedJetONNXJetTagsProducer
ak8ScoutingJetParticleNetJetTags = cms.EDProducer("BoostedJetONNXJetTagsProducer",
      jets = cms.InputTag("ak8ScoutingJets"),
      produceValueMap = cms.untracked.bool(True),
      src = cms.InputTag("ak8ScoutingJetParticleNetJetTagInfos"),
      preprocess_json = cms.string("RecoBTag/Combined/data/Run3Scouting/ParticleNetAK8/General/V00/preprocess.json"),
      model_path = cms.FileInPath("RecoBTag/Combined/data/Run3Scouting/ParticleNetAK8/General/V00/particle-net.onnx"),
      flav_names = cms.vstring(["probHbb", "probHcc","probHqq", "probQCDall"]),
      debugMode = cms.untracked.bool(False),
  )

ak8ScoutingJetParticleNetMassRegressionJetTags = cms.EDProducer("BoostedJetONNXJetTagsProducer",
      jets = cms.InputTag("ak8ScoutingJets"),
      produceValueMap = cms.untracked.bool(True),
      src = cms.InputTag("ak8ScoutingJetParticleNetJetTagInfos"),
      preprocess_json = cms.string("RecoBTag/Combined/data/Run3Scouting/ParticleNetAK8/MassRegression/V00/preprocess.json"),
      model_path = cms.FileInPath("RecoBTag/Combined/data/Run3Scouting/ParticleNetAK8/MassRegression/V00/particle-net.onnx"),
      flav_names = cms.vstring(["mass"]),
      debugMode = cms.untracked.bool(False),
  )

ak8ScoutingJetTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
      src = cms.InputTag("ak8ScoutingJets"),
      name = cms.string("ScoutingFatJet"),
      cut = cms.string(""),
      doc = cms.string("ScoutingFatJet"),
      singleton = cms.bool(False),
      extension = cms.bool(False), # this is the main table
      externalVariables = cms.PSet(
         #genJetAK8Idx = ExtVar(cms.InputTag("ak8ScoutingJetMatchGen"), int, doc="gen jet idx"),
         msoftdrop = ExtVar(cms.InputTag('ak8ScoutingJetsSoftDropMass'), float, doc="Softdrop mass", precision=10),
         n2b1 = ExtVar(cms.InputTag('ak8ScoutingJetEcfNbeta1:ecfN2'), float, doc="N2 with beta=1", precision=10),
         n3b1 = ExtVar(cms.InputTag('ak8ScoutingJetEcfNbeta1:ecfN3'), float, doc="N3 with beta=1", precision=10),
         tau1 = ExtVar(cms.InputTag('ak8ScoutingJetNjettiness:tau1'), float, doc="Nsubjettiness (1 axis)", precision=10),
         tau2 = ExtVar(cms.InputTag('ak8ScoutingJetNjettiness:tau2'), float, doc="Nsubjettiness (2 axis)", precision=10),
         tau3 = ExtVar(cms.InputTag('ak8ScoutingJetNjettiness:tau3'), float, doc="Nsubjettiness (3 axis)", precision=10),
         tau4 = ExtVar(cms.InputTag('ak8ScoutingJetNjettiness:tau4'), float, doc="Nsubjettiness (4 axis)", precision=10),
         particleNet_mass = ExtVar(cms.InputTag('ak8ScoutingJetParticleNetMassRegressionJetTags:mass'), float, doc="ParticleNet regressed mass", precision=10),
         particleNet_prob_Hbb = ExtVar(cms.InputTag('ak8ScoutingJetParticleNetJetTags:probHbb'), float, doc="ParticleNet probability of Hbb", precision=10),
         particleNet_prob_Hcc = ExtVar(cms.InputTag('ak8ScoutingJetParticleNetJetTags:probHcc'), float, doc="ParticleNet probability of Hcc", precision=10),
         particleNet_prob_Hqq = ExtVar(cms.InputTag('ak8ScoutingJetParticleNetJetTags:probHqq'), float, doc="ParticleNet probability of Hqq", precision=10),
         particleNet_prob_QCD = ExtVar(cms.InputTag('ak8ScoutingJetParticleNetJetTags:probQCDall'), float, doc="ParticleNet probability of QCD", precision=10),
      ),
      variables = cms.PSet(
         P4Vars,
         area = Var("jetArea()", float, doc="jet catchment area, for JECs",precision=10),
         chHEF = Var("chargedHadronEnergy()/(chargedHadronEnergy()+neutralHadronEnergy()+photonEnergy()+electronEnergy()+muonEnergy())", float, doc="charged Hadron Energy Fraction", precision= 6),
         neHEF = Var("neutralHadronEnergy()/(chargedHadronEnergy()+neutralHadronEnergy()+photonEnergy()+electronEnergy()+muonEnergy())", float, doc="neutral Hadron Energy Fraction", precision= 6),
         chEmEF = Var("(electronEnergy()+muonEnergy())/(chargedHadronEnergy()+neutralHadronEnergy()+photonEnergy()+electronEnergy()+muonEnergy())", float, doc="charged Electromagnetic Energy Fraction", precision= 6),
         neEmEF = Var("(photonEnergy())/(chargedHadronEnergy()+neutralHadronEnergy()+photonEnergy()+electronEnergy()+muonEnergy())", float, doc="neutral Electromagnetic Energy Fraction", precision= 6),
         muEF = Var("(muonEnergy())/(chargedHadronEnergy()+neutralHadronEnergy()+photonEnergy()+electronEnergy()+muonEnergy())", float, doc="muon Energy Fraction", precision= 6),
         nCh = Var("chargedHadronMultiplicity()", int, doc="number of charged hadrons in the jet"),
         nNh = Var("neutralHadronMultiplicity()", int, doc="number of neutral hadrons in the jet"),
         nMuons = Var("muonMultiplicity()", int, doc="number of muons in the jet"),
         nElectrons = Var("electronMultiplicity()", int, doc="number of electrons in the jet"),
         nPhotons = Var("photonMultiplicity()", int, doc="number of photons in the jet"),
         nConstituents = Var("numberOfDaughters()", "uint8", doc="number of particles in the jet")
      ),
  )

ak8ScoutingJetMatchGen = cms.EDProducer("RecoJetToGenJetDeltaRValueMapProducer",
      src = cms.InputTag("ak8ScoutingJets"),
      matched = cms.InputTag("slimmedGenJetsAK8"),
      distMax = cms.double(0.8),
      value = cms.string("index"),
  )

ak8ScoutingJetExtTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
      src = cms.InputTag("ak8ScoutingJets"),
      name = cms.string("ScoutingFatJet"),
      cut = cms.string(""),
      singleton = cms.bool(False),
      extension = cms.bool(True),
      externalVariables = cms.PSet(
         genJetAK8Idx = ExtVar(cms.InputTag("ak8ScoutingJetMatchGen"), int, doc="gen jet idx"),
      ),
      variables = cms.PSet(),
  )
