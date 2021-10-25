#include "DQMOffline/EGamma/plugins/ElectronTagProbeAnalyzer.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "TMath.h"

#include <iostream>
#include <vector>

using namespace reco;

ElectronTagProbeAnalyzer::ElectronTagProbeAnalyzer(const edm::ParameterSet& conf) : ElectronDqmAnalyzerBase(conf) {
  // general, collections
  Selection_ = conf.getParameter<int>("Selection");
  electronCollection_ = consumes<GsfElectronCollection>(conf.getParameter<edm::InputTag>("ElectronCollection"));
  matchingObjectCollection_ =
      consumes<SuperClusterCollection>(conf.getParameter<edm::InputTag>("MatchingObjectCollection"));
  trackCollection_ = consumes<TrackCollection>(conf.getParameter<edm::InputTag>("TrackCollection"));
  vertexCollection_ = consumes<VertexCollection>(conf.getParameter<edm::InputTag>("VertexCollection"));
  gsftrackCollection_ = consumes<GsfTrackCollection>(conf.getParameter<edm::InputTag>("GsfTrackCollection"));
  beamSpotTag_ = consumes<BeamSpot>(conf.getParameter<edm::InputTag>("BeamSpot"));
  readAOD_ = conf.getParameter<bool>("ReadAOD");

  // tag and probe
  massLow_ = conf.getParameter<double>("MassLow");
  massHigh_ = conf.getParameter<double>("MassHigh");
  TPchecksign_ = conf.getParameter<bool>("TpCheckSign");
  TAGcheckclass_ = conf.getParameter<bool>("TagCheckClass");
  PROBEetcut_ = conf.getParameter<bool>("ProbeEtCut");
  PROBEcheckclass_ = conf.getParameter<bool>("ProbeCheckClass");

  // electron selection
  minEt_ = conf.getParameter<double>("MinEt");
  minPt_ = conf.getParameter<double>("MinPt");
  maxAbsEta_ = conf.getParameter<double>("MaxAbsEta");
  isEB_ = conf.getParameter<bool>("SelectEb");
  isEE_ = conf.getParameter<bool>("SelectEe");
  isNotEBEEGap_ = conf.getParameter<bool>("SelectNotEbEeGap");
  isEcalDriven_ = conf.getParameter<bool>("SelectEcalDriven");
  isTrackerDriven_ = conf.getParameter<bool>("SelectTrackerDriven");
  eOverPMinBarrel_ = conf.getParameter<double>("MinEopBarrel");
  eOverPMaxBarrel_ = conf.getParameter<double>("MaxEopBarrel");
  eOverPMinEndcaps_ = conf.getParameter<double>("MinEopEndcaps");
  eOverPMaxEndcaps_ = conf.getParameter<double>("MaxEopEndcaps");
  dEtaMinBarrel_ = conf.getParameter<double>("MinDetaBarrel");
  dEtaMaxBarrel_ = conf.getParameter<double>("MaxDetaBarrel");
  dEtaMinEndcaps_ = conf.getParameter<double>("MinDetaEndcaps");
  dEtaMaxEndcaps_ = conf.getParameter<double>("MaxDetaEndcaps");
  dPhiMinBarrel_ = conf.getParameter<double>("MinDphiBarrel");
  dPhiMaxBarrel_ = conf.getParameter<double>("MaxDphiBarrel");
  dPhiMinEndcaps_ = conf.getParameter<double>("MinDphiEndcaps");
  dPhiMaxEndcaps_ = conf.getParameter<double>("MaxDphiEndcaps");
  sigIetaIetaMinBarrel_ = conf.getParameter<double>("MinSigIetaIetaBarrel");
  sigIetaIetaMaxBarrel_ = conf.getParameter<double>("MaxSigIetaIetaBarrel");
  sigIetaIetaMinEndcaps_ = conf.getParameter<double>("MinSigIetaIetaEndcaps");
  sigIetaIetaMaxEndcaps_ = conf.getParameter<double>("MaxSigIetaIetaEndcaps");
  hadronicOverEmMaxBarrel_ = conf.getParameter<double>("MaxHoeBarrel");
  hadronicOverEmMaxEndcaps_ = conf.getParameter<double>("MaxHoeEndcaps");
  mvaMin_ = conf.getParameter<double>("MinMva");
  tipMaxBarrel_ = conf.getParameter<double>("MaxTipBarrel");
  tipMaxEndcaps_ = conf.getParameter<double>("MaxTipEndcaps");
  tkIso03Max_ = conf.getParameter<double>("MaxTkIso03");
  hcalIso03Depth1MaxBarrel_ = conf.getParameter<double>("MaxHcalIso03Depth1Barrel");
  hcalIso03Depth1MaxEndcaps_ = conf.getParameter<double>("MaxHcalIso03Depth1Endcaps");
  hcalIso03Depth2MaxEndcaps_ = conf.getParameter<double>("MaxHcalIso03Depth2Endcaps");
  ecalIso03MaxBarrel_ = conf.getParameter<double>("MaxEcalIso03Barrel");
  ecalIso03MaxEndcaps_ = conf.getParameter<double>("MaxEcalIso03Endcaps");

  // for trigger
  triggerResults_ = conf.getParameter<edm::InputTag>("TriggerResults");
  //  HLTPathsByName_= conf.getParameter<std::vector<std::string > >("HltPaths");
  //  HLTPathsByIndex_.resize(HLTPathsByName_.size());

  // histos limits and binning
  nbineta = conf.getParameter<int>("NbinEta");
  nbineta2D = conf.getParameter<int>("NbinEta2D");
  etamin = conf.getParameter<double>("EtaMin");
  etamax = conf.getParameter<double>("EtaMax");
  //
  nbinphi = conf.getParameter<int>("NbinPhi");
  nbinphi2D = conf.getParameter<int>("NbinPhi2D");
  phimin = conf.getParameter<double>("PhiMin");
  phimax = conf.getParameter<double>("PhiMax");
  //
  nbinpt = conf.getParameter<int>("NbinPt");
  nbinpteff = conf.getParameter<int>("NbinPtEff");
  nbinpt2D = conf.getParameter<int>("NbinPt2D");
  ptmax = conf.getParameter<double>("PtMax");
  //
  nbinp = conf.getParameter<int>("NbinP");
  nbinp2D = conf.getParameter<int>("NbinP2D");
  pmax = conf.getParameter<double>("PMax");
  //
  nbineop = conf.getParameter<int>("NbinEop");
  nbineop2D = conf.getParameter<int>("NbinEop2D");
  eopmax = conf.getParameter<double>("EopMax");
  eopmaxsht = conf.getParameter<double>("EopMaxSht");
  //
  nbindeta = conf.getParameter<int>("NbinDeta");
  detamin = conf.getParameter<double>("DetaMin");
  detamax = conf.getParameter<double>("DetaMax");
  //
  nbindphi = conf.getParameter<int>("NbinDphi");
  dphimin = conf.getParameter<double>("DphiMin");
  dphimax = conf.getParameter<double>("DphiMax");
  //
  nbindetamatch = conf.getParameter<int>("NbinDetaMatch");
  nbindetamatch2D = conf.getParameter<int>("NbinDetaMatch2D");
  detamatchmin = conf.getParameter<double>("DetaMatchMin");
  detamatchmax = conf.getParameter<double>("DetaMatchMax");
  //
  nbindphimatch = conf.getParameter<int>("NbinDphiMatch");
  nbindphimatch2D = conf.getParameter<int>("NbinDphiMatch2D");
  dphimatchmin = conf.getParameter<double>("DphiMatchMin");
  dphimatchmax = conf.getParameter<double>("DphiMatchMax");
  //
  nbinfhits = conf.getParameter<int>("NbinFhits");
  fhitsmax = conf.getParameter<double>("FhitsMax");
  //
  nbinlhits = conf.getParameter<int>("NbinLhits");
  lhitsmax = conf.getParameter<double>("LhitsMax");
  //
  nbinxyz = conf.getParameter<int>("NbinXyz");
  nbinxyz2D = conf.getParameter<int>("NbinXyz2D");
  //
  nbinpoptrue = conf.getParameter<int>("NbinPopTrue");
  poptruemin = conf.getParameter<double>("PopTrueMin");
  poptruemax = conf.getParameter<double>("PopTrueMax");
  //
  nbinmee = conf.getParameter<int>("NbinMee");
  meemin = conf.getParameter<double>("MeeMin");
  meemax = conf.getParameter<double>("MeeMax");
  //
  nbinhoe = conf.getParameter<int>("NbinHoe");
  hoemin = conf.getParameter<double>("HoeMin");
  hoemax = conf.getParameter<double>("HoeMax");
}

ElectronTagProbeAnalyzer::~ElectronTagProbeAnalyzer() {}

void ElectronTagProbeAnalyzer::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const&, edm::EventSetup const&) {
  iBooker.setCurrentFolder(outputInternalPath_);
  nEvents_ = 0;
  //nAfterTrigger_ = 0 ;

  // basic quantities
  h1_vertexPt_barrel = bookH1(
      iBooker, "vertexPt_barrel", "ele transverse momentum in barrel", nbinpt, 0., ptmax, "p_{T vertex} (GeV/c)");
  h1_vertexPt_endcaps = bookH1(
      iBooker, "vertexPt_endcaps", "ele transverse momentum in endcaps", nbinpt, 0., ptmax, "p_{T vertex} (GeV/c)");
  h1_vertexEta = bookH1(iBooker, "vertexEta", "ele momentum #eta", nbineta, etamin, etamax, "#eta");
  h2_vertexEtaVsPhi = bookH2(iBooker,
                             "vertexEtaVsPhi",
                             "ele momentum #eta vs #phi",
                             nbineta2D,
                             etamin,
                             etamax,
                             nbinphi2D,
                             phimin,
                             phimax,
                             "#eta",
                             "#phi (rad)");
  h2_vertexXvsY = bookH2(
      iBooker, "vertexXvsY", "ele vertex x vs y", nbinxyz2D, -0.1, 0.1, nbinxyz2D, -0.1, 0.1, "x (cm)", "y (cm)");
  h1_vertexZ = bookH1(iBooker, "vertexZ", "ele vertex z", nbinxyz, -25, 25, "z (cm)");

  // super-clusters
  //  h1_sclPhi = bookH1(iBooker, "sclPhi","ele supercluster phi",nbinphi,phimin,phimax);
  h1_sclEt = bookH1(iBooker, "sclEt", "ele supercluster transverse energy", nbinpt, 0., ptmax);

  // electron track
  h1_chi2 = bookH1(iBooker, "chi2", "ele track #chi^{2}", 100, 0., 15., "#Chi^{2}");
  h1_foundHits = bookH1(iBooker, "foundHits", "ele track # found hits", nbinfhits, 0., fhitsmax, "N_{hits}");
  h1_lostHits = bookH1(iBooker, "lostHits", "ele track # lost hits", 5, 0., 5., "N_{lost hits}");

  // electron matching and ID
  h1_Eop_barrel = bookH1(iBooker, "Eop_barrel", "ele E/P_{vertex} in barrel", nbineop, 0., eopmax, "E/P_{vertex}");
  h1_Eop_endcaps = bookH1(iBooker, "Eop_endcaps", "ele E/P_{vertex} in endcaps", nbineop, 0., eopmax, "E/P_{vertex}");
  h1_EeleOPout_barrel =
      bookH1(iBooker, "EeleOPout_barrel", "ele E_{ele}/P_{out} in barrel", nbineop, 0., eopmax, "E_{ele}/P_{out}");
  h1_EeleOPout_endcaps =
      bookH1(iBooker, "EeleOPout_endcaps", "ele E_{ele}/P_{out} in endcaps", nbineop, 0., eopmax, "E_{ele}/P_{out}");
  h1_dEtaSc_propVtx_barrel = bookH1(iBooker,
                                    "dEtaSc_propVtx_barrel",
                                    "ele #eta_{sc} - #eta_{tr}, prop from vertex, in barrel",
                                    nbindetamatch,
                                    detamatchmin,
                                    detamatchmax,
                                    "#eta_{sc} - #eta_{tr}");
  h1_dEtaSc_propVtx_endcaps = bookH1(iBooker,
                                     "dEtaSc_propVtx_endcaps",
                                     "ele #eta_{sc} - #eta_{tr}, prop from vertex, in endcaps",
                                     nbindetamatch,
                                     detamatchmin,
                                     detamatchmax,
                                     "#eta_{sc} - #eta_{tr}");
  h1_dEtaEleCl_propOut_barrel = bookH1(iBooker,
                                       "dEtaEleCl_propOut_barrel",
                                       "ele #eta_{EleCl} - #eta_{tr}, prop from outermost, in barrel",
                                       nbindetamatch,
                                       detamatchmin,
                                       detamatchmax,
                                       "#eta_{elecl} - #eta_{tr}");
  h1_dEtaEleCl_propOut_endcaps = bookH1(iBooker,
                                        "dEtaEleCl_propOut_endcaps",
                                        "ele #eta_{EleCl} - #eta_{tr}, prop from outermost, in endcaps",
                                        nbindetamatch,
                                        detamatchmin,
                                        detamatchmax,
                                        "#eta_{elecl} - #eta_{tr}");
  h1_dPhiSc_propVtx_barrel = bookH1(iBooker,
                                    "dPhiSc_propVtx_barrel",
                                    "ele #phi_{sc} - #phi_{tr}, prop from vertex, in barrel",
                                    nbindphimatch,
                                    dphimatchmin,
                                    dphimatchmax,
                                    "#phi_{sc} - #phi_{tr} (rad)");
  h1_dPhiSc_propVtx_endcaps = bookH1(iBooker,
                                     "dPhiSc_propVtx_endcaps",
                                     "ele #phi_{sc} - #phi_{tr}, prop from vertex, in endcaps",
                                     nbindphimatch,
                                     dphimatchmin,
                                     dphimatchmax,
                                     "#phi_{sc} - #phi_{tr} (rad)");
  h1_dPhiEleCl_propOut_barrel = bookH1(iBooker,
                                       "dPhiEleCl_propOut_barrel",
                                       "ele #phi_{EleCl} - #phi_{tr}, prop from outermost, in barrel",
                                       nbindphimatch,
                                       dphimatchmin,
                                       dphimatchmax,
                                       "#phi_{elecl} - #phi_{tr} (rad)");
  h1_dPhiEleCl_propOut_endcaps = bookH1(iBooker,
                                        "dPhiEleCl_propOut_endcaps",
                                        "ele #phi_{EleCl} - #phi_{tr}, prop from outermost, in endcaps",
                                        nbindphimatch,
                                        dphimatchmin,
                                        dphimatchmax,
                                        "#phi_{elecl} - #phi_{tr} (rad)");
  h1_Hoe_barrel = bookH1(iBooker,
                         "Hoe_barrel",
                         "ele hadronic energy / em energy, in barrel",
                         nbinhoe,
                         hoemin,
                         hoemax,
                         "H/E",
                         "Events",
                         "ELE_LOGY E1 P");
  h1_Hoe_endcaps = bookH1(iBooker,
                          "Hoe_endcaps",
                          "ele hadronic energy / em energy, in endcaps",
                          nbinhoe,
                          hoemin,
                          hoemax,
                          "H/E",
                          "Events",
                          "ELE_LOGY E1 P");
  h1_sclSigEtaEta_barrel =
      bookH1(iBooker, "sclSigEtaEta_barrel", "ele supercluster sigma eta eta in barrel", 100, 0., 0.05);
  h1_sclSigEtaEta_endcaps =
      bookH1(iBooker, "sclSigEtaEta_endcaps", "ele supercluster sigma eta eta in endcaps", 100, 0., 0.05);

  // fbrem
  h1_fbrem = bookH1(iBooker, "fbrem", "ele brem fraction", 100, 0., 1., "P_{in} - P_{out} / P_{in}");
  h1_classes = bookH1(iBooker, "classes", "ele electron classes", 10, 0.0, 10.);

  // pflow
  h1_mva = bookH1(iBooker, "mva", "ele identification mva", 100, -1., 1.);
  h1_provenance = bookH1(iBooker, "provenance", "ele provenance", 5, -2., 3.);

  // isolation
  h1_tkSumPt_dr03 = bookH1(iBooker,
                           "tkSumPt_dr03",
                           "tk isolation sum, dR=0.3",
                           100,
                           0.0,
                           20.,
                           "TkIsoSum, cone 0.3 (GeV/c)",
                           "Events",
                           "ELE_LOGY E1 P");
  h1_ecalRecHitSumEt_dr03 = bookH1(iBooker,
                                   "ecalRecHitSumEt_dr03",
                                   "ecal isolation sum, dR=0.3",
                                   100,
                                   0.0,
                                   20.,
                                   "EcalIsoSum, cone 0.3 (GeV)",
                                   "Events",
                                   "ELE_LOGY E1 P");
  h1_hcalTowerSumEt_dr03 = bookH1(iBooker,
                                  "hcalTowerSumEt_dr03",
                                  "hcal isolation sum, dR=0.3",
                                  100,
                                  0.0,
                                  20.,
                                  "HcalIsoSum, cone 0.3 (GeV)",
                                  "Events",
                                  "ELE_LOGY E1 P");

  // di-electron mass
  setBookIndex(200);
  h1_mee = bookH1(iBooker, "mesc", "Tag ele Probe SC invariant mass", nbinmee, meemin, meemax, "m_{eSC} (GeV/c^{2})");
  h1_mee_os = bookH1(iBooker,
                     "mee_os",
                     "ele pairs invariant mass, opposite sign",
                     nbinmee,
                     meemin,
                     meemax,
                     "m_{e^{+}e^{-}} (GeV/c^{2})");

  //===========================
  // histos for matching and matched matched objects
  //===========================

  // matching object
  std::string matchingObjectType;
  Labels l;
  labelsForToken(matchingObjectCollection_, l);
  if (std::string::npos != std::string(l.module).find("SuperCluster", 0)) {
    matchingObjectType = "SC";
  }
  if (matchingObjectType.empty()) {
    edm::LogError("ElectronMcFakeValidator::beginJob") << "Unknown matching object type !";
  } else {
    edm::LogInfo("ElectronMcFakeValidator::beginJob") << "Matching object type: " << matchingObjectType;
  }

  // matching object distributions
  h1_matchingObject_Eta = bookH1withSumw2(
      iBooker, "matchingObject_Eta", matchingObjectType + " #eta", nbineta, etamin, etamax, "#eta_{SC}");
  h1_matchingObject_Pt =
      bookH1withSumw2(iBooker, "matchingObject_Pt", matchingObjectType + " pt", nbinpteff, 5., ptmax);
  h1_matchingObject_Phi =
      bookH1withSumw2(iBooker, "matchingObject_Phi", matchingObjectType + " phi", nbinphi, phimin, phimax);
  //h1_matchingObject_Z = bookH1withSumw2(iBooker, "matchingObject_Z",matchingObjectType+" z",nbinxyz,-25,25);

  h1_matchedObject_Eta =
      bookH1withSumw2(iBooker, "matchedObject_Eta", "Efficiency vs matching SC #eta", nbineta, etamin, etamax);
  h1_matchedObject_Pt =
      bookH1withSumw2(iBooker, "matchedObject_Pt", "Efficiency vs matching SC E_{T}", nbinpteff, 5., ptmax);
  h1_matchedObject_Phi =
      bookH1withSumw2(iBooker, "matchedObject_Phi", "Efficiency vs matching SC phi", nbinphi, phimin, phimax);
  //h1_matchedObject_Z = bookH1withSumw2(iBooker, "matchedObject_Z","Efficiency vs matching SC z",nbinxyz,-25,25);
}

void ElectronTagProbeAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  nEvents_++;

  edm::Handle<GsfElectronCollection> gsfElectrons;
  iEvent.getByToken(electronCollection_, gsfElectrons);
  edm::Handle<reco::SuperClusterCollection> recoClusters;
  iEvent.getByToken(matchingObjectCollection_, recoClusters);
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByToken(trackCollection_, tracks);
  edm::Handle<reco::GsfTrackCollection> gsfTracks;
  iEvent.getByToken(gsftrackCollection_, gsfTracks);
  edm::Handle<reco::VertexCollection> vertices;
  iEvent.getByToken(vertexCollection_, vertices);
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByToken(beamSpotTag_, recoBeamSpotHandle);
  const BeamSpot bs = *recoBeamSpotHandle;

  edm::EventNumber_t ievt = iEvent.id().event();
  edm::RunNumber_t irun = iEvent.id().run();
  edm::LuminosityBlockNumber_t ils = iEvent.luminosityBlock();

  edm::LogInfo("ElectronMcSignalValidator::analyze")
      << "Treating " << gsfElectrons.product()->size() << " electrons"
      << " from event " << ievt << " in run " << irun << " and lumiblock " << ils;
  //h1_num_->Fill((*gsfElectrons).size()) ;

  std::vector<std::pair<double, double> > TTCheck;
  std::vector<std::pair<double, double> > TTscCheck;

  // selected rec electrons
  reco::GsfElectronCollection::const_iterator gsfIter;
  for (gsfIter = gsfElectrons->begin(); gsfIter != gsfElectrons->end(); gsfIter++) {
    // vertex TIP
    double vertexTIP = (gsfIter->vertex().x() - bs.position().x()) * (gsfIter->vertex().x() - bs.position().x()) +
                       (gsfIter->vertex().y() - bs.position().y()) * (gsfIter->vertex().y() - bs.position().y());
    vertexTIP = sqrt(vertexTIP);

    // select electrons
    if (!selected(gsfIter, vertexTIP))
      continue;

    reco::SuperClusterRef sclTagRef = gsfIter->superCluster();
    reco::SuperClusterCollection::const_iterator moIter;
    for (moIter = recoClusters->begin(); moIter != recoClusters->end(); moIter++) {
      if (moIter->eta() == sclTagRef->eta())
        continue;

      /*
      if
        ( moIter->energy()/cosh(moIter->eta())>maxPtMatchingObject_ ||
          std::abs(moIter->eta())> maxAbsEtaMatchingObject_ )
        { continue ; }
      */

      // Additional cuts on Tag

      // Additional cuts on Probe
      if (PROBEetcut_ && (moIter->energy() / cosh(moIter->eta()) < minEt_))
        continue;

      float SCenergy = moIter->energy();
      math::XYZPoint caloposition = moIter->position();
      float theta = caloposition.Theta();
      math::XYZVector momentum;
      float SCmomentumX = SCenergy * sin(theta) * cos(moIter->phi());
      float SCmomentumY = SCenergy * sin(theta) * sin(moIter->phi());
      float SCmomentumZ = SCenergy * cos(theta);
      const reco::Particle::LorentzVector pSCprobecandidate(SCmomentumX, SCmomentumY, SCmomentumZ, SCenergy);

      math::XYZTLorentzVector p12 = (*gsfIter).p4() + pSCprobecandidate;
      float mee2 = p12.Dot(p12);
      float invMass = mee2 > 0. ? sqrt(mee2) : 0.;

      if (invMass < massLow_ || invMass > massHigh_)
        continue;

      h1_matchingObject_Eta->Fill(moIter->eta());
      h1_matchingObject_Pt->Fill(moIter->energy() / cosh(moIter->eta()));
      h1_matchingObject_Phi->Fill(moIter->phi());
      //h1_matchingObject_Z->Fill(  moIter->z() );

      reco::GsfElectron bestGsfElectron;
      reco::SuperClusterRef sclProbeRef;
      bool okGsfFound = false;
      //double gsfOkRatio = 999999.;
      reco::GsfElectronCollection::const_iterator gsfIter2;

      for (gsfIter2 = gsfElectrons->begin(); gsfIter2 != gsfElectrons->end(); gsfIter2++) {
        // matching with ref
        sclProbeRef = gsfIter2->superCluster();

        if (sclProbeRef->eta() == moIter->eta()) {
          //std::cout << "les deux ref SC sont egales : " << std::endl;
          bestGsfElectron = *gsfIter2;

          // opposite sign checking
          bool opsign = (((gsfIter->charge()) * (bestGsfElectron.charge())) < 0.);
          if (TPchecksign_ && !opsign) {
            okGsfFound = false;
            h1_mee->Fill(invMass);
            break;
          } else {
            okGsfFound = true;
          }
        }  //fi on gsfEleSC.eta == probeSC.eta

        //        // matching with cone
        //        double dphi = gsfIter2->phi()-moIter->phi();
        //        if (std::abs(dphi)>CLHEP::pi)
        //         dphi = dphi < 0? (CLHEP::twopi) + dphi : dphi - CLHEP::twopi;
        //        double deltaR = sqrt(pow((moIter->eta()-gsfIter2->eta()),2) + pow(dphi,2));
        //        if ( deltaR < deltaR_ )
        //        {
        //         double tmpGsfRatio = gsfIter2->p()/moIter->energy();
        //         if ( std::abs(tmpGsfRatio-1) < std::abs(gsfOkRatio-1))
        //          {
        //           gsfOkRatio = tmpGsfRatio;
        //           bestGsfElectron=*gsfIter2;
        //           okGsfFound = true;
        //          }
        //        } // fi on deltaR

      }  // end of loop on gsfEle to find the best one which matches with probe SC

      if (okGsfFound) {
        // fill matched histos for eff
        fillMatchedHistos(moIter, bestGsfElectron);

        // basic quantities
        if (bestGsfElectron.isEB())
          h1_vertexPt_barrel->Fill(bestGsfElectron.pt());
        if (bestGsfElectron.isEE())
          h1_vertexPt_endcaps->Fill(bestGsfElectron.pt());
        h1_vertexEta->Fill(bestGsfElectron.eta());
        h2_vertexEtaVsPhi->Fill(bestGsfElectron.eta(), bestGsfElectron.phi());
        h2_vertexXvsY->Fill(bestGsfElectron.vertex().x(), bestGsfElectron.vertex().y());
        h1_vertexZ->Fill(bestGsfElectron.vertex().z());

        // supercluster related distributions
        reco::SuperClusterRef sclRef = bestGsfElectron.superCluster();
        double R = TMath::Sqrt(sclRef->x() * sclRef->x() + sclRef->y() * sclRef->y() + sclRef->z() * sclRef->z());
        double Rt = TMath::Sqrt(sclRef->x() * sclRef->x() + sclRef->y() * sclRef->y());
        h1_sclEt->Fill(sclRef->energy() * (Rt / R));

        if (!readAOD_) {  // track extra does not exist in AOD
          h1_foundHits->Fill(bestGsfElectron.gsfTrack()->numberOfValidHits());
          h1_lostHits->Fill(bestGsfElectron.gsfTrack()->numberOfLostHits());
          h1_chi2->Fill(bestGsfElectron.gsfTrack()->normalizedChi2());
        }

        // match distributions
        if (bestGsfElectron.isEB()) {
          h1_Eop_barrel->Fill(bestGsfElectron.eSuperClusterOverP());
          h1_EeleOPout_barrel->Fill(bestGsfElectron.eEleClusterOverPout());
          h1_dEtaSc_propVtx_barrel->Fill(bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
          h1_dEtaEleCl_propOut_barrel->Fill(bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
          h1_dPhiSc_propVtx_barrel->Fill(bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
          h1_dPhiEleCl_propOut_barrel->Fill(bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
          h1_Hoe_barrel->Fill(bestGsfElectron.hadronicOverEm());
          h1_sclSigEtaEta_barrel->Fill(bestGsfElectron.scSigmaEtaEta());
        }
        if (bestGsfElectron.isEE()) {
          h1_Eop_endcaps->Fill(bestGsfElectron.eSuperClusterOverP());
          h1_EeleOPout_endcaps->Fill(bestGsfElectron.eEleClusterOverPout());
          h1_dEtaSc_propVtx_endcaps->Fill(bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
          h1_dEtaEleCl_propOut_endcaps->Fill(bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
          h1_dPhiSc_propVtx_endcaps->Fill(bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
          h1_dPhiEleCl_propOut_endcaps->Fill(bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
          h1_Hoe_endcaps->Fill(bestGsfElectron.hadronicOverEm());
          h1_sclSigEtaEta_endcaps->Fill(bestGsfElectron.scSigmaEtaEta());
        }

        // fbrem
        h1_fbrem->Fill(bestGsfElectron.fbrem());
        int eleClass = bestGsfElectron.classification();
        if (bestGsfElectron.isEE())
          eleClass += 5;
        h1_classes->Fill(eleClass);

        // pflow
        h1_mva->Fill(bestGsfElectron.mva_e_pi());
        if (bestGsfElectron.ecalDrivenSeed())
          h1_provenance->Fill(1.);
        if (bestGsfElectron.trackerDrivenSeed())
          h1_provenance->Fill(-1.);
        if (bestGsfElectron.trackerDrivenSeed() || bestGsfElectron.ecalDrivenSeed())
          h1_provenance->Fill(0.);
        if (bestGsfElectron.trackerDrivenSeed() && !bestGsfElectron.ecalDrivenSeed())
          h1_provenance->Fill(-2.);
        if (!bestGsfElectron.trackerDrivenSeed() && bestGsfElectron.ecalDrivenSeed())
          h1_provenance->Fill(2.);

        // isolation
        h1_tkSumPt_dr03->Fill(bestGsfElectron.dr03TkSumPt());
        h1_ecalRecHitSumEt_dr03->Fill(bestGsfElectron.dr03EcalRecHitSumEt());
        h1_hcalTowerSumEt_dr03->Fill(bestGsfElectron.dr03HcalTowerSumEt());

        // inv Mass with opposite sign
        bool invMassTTAlreadyFilled = false;
        if (!TTCheck.empty()) {
          int TTCheckDim = TTCheck.size();
          for (int i = 0; i < TTCheckDim; i++) {
            if ((bestGsfElectron.eta() == TTCheck.at(i).first) && (gsfIter->eta() == TTCheck.at(i).second)) {
              invMassTTAlreadyFilled = true;
            }
          }
        }

        if (!invMassTTAlreadyFilled && (((gsfIter->charge()) * (bestGsfElectron.charge())) < 0.)) {
          h1_mee->Fill(invMass);
          math::XYZTLorentzVector p12bis = (*gsfIter).p4() + bestGsfElectron.p4();
          float mee2bis = p12.Dot(p12bis);
          float invMassEE = mee2bis > 0. ? sqrt(mee2bis) : 0.;
          if (invMassEE >= massLow_ && invMassEE <= massHigh_) {
            h1_mee_os->Fill(invMassEE);
          }
          std::pair<double, double> p(gsfIter->eta(), bestGsfElectron.eta());
          TTCheck.push_back(p);
        }

      }  // fi on OkGsfFound
      else {
        h1_mee->Fill(invMass);
      }

    }  // end of loop on SC to find probe SC

  }  // end of loop on Tag gsfEle
}

float ElectronTagProbeAnalyzer::computeInvMass(const reco::GsfElectron& e1, const reco::GsfElectron& e2) {
  math::XYZTLorentzVector p12 = e1.p4() + e2.p4();
  float mee2 = p12.Dot(p12);
  float invMass = mee2 > 0. ? sqrt(mee2) : 0.;
  return invMass;
}

void ElectronTagProbeAnalyzer::fillMatchedHistos(const reco::SuperClusterCollection::const_iterator& moIter,
                                                 const reco::GsfElectron& electron) {
  // generated distributions for matched electrons
  h1_matchedObject_Eta->Fill(moIter->eta());
  //  h1_matchedObject_AbsEta->Fill( std::abs(moIter->eta()) );
  h1_matchedObject_Pt->Fill(moIter->energy() / cosh(moIter->eta()));
  h1_matchedObject_Phi->Fill(moIter->phi());
  //h1_matchedObject_Z->Fill( moIter->z() );

  //classes
  //  int eleClass = electron.classification() ;
  //  h_classes->Fill(eleClass) ;
  //  h_matchedEle_eta->Fill(std::abs(electron.eta()));
  //  if (electron.classification() == GsfElectron::GOLDEN) h_matchedEle_eta_golden->Fill(std::abs(electron.eta()));
  //  if (electron.classification() == GsfElectron::SHOWERING) h_matchedEle_eta_shower->Fill(std::abs(electron.eta()));
  //  //if (electron.classification() == GsfElectron::BIGBREM) h_matchedEle_eta_bbrem->Fill(std::abs(electron.eta()));
  //  //if (electron.classification() == GsfElectron::OLDNARROW) h_matchedEle_eta_narrow->Fill(std::abs(electron.eta()));
}

bool ElectronTagProbeAnalyzer::selected(const reco::GsfElectronCollection::const_iterator& gsfIter, double vertexTIP) {
  if ((Selection_ > 0) && generalCut(gsfIter))
    return false;
  if ((Selection_ >= 1) && etCut(gsfIter))
    return false;
  if ((Selection_ >= 2) && isolationCut(gsfIter, vertexTIP))
    return false;
  if ((Selection_ >= 3) && idCut(gsfIter))
    return false;
  return true;
}

bool ElectronTagProbeAnalyzer::generalCut(const reco::GsfElectronCollection::const_iterator& gsfIter) {
  if (std::abs(gsfIter->eta()) > maxAbsEta_)
    return true;
  if (gsfIter->pt() < minPt_)
    return true;

  if (gsfIter->isEB() && isEE_)
    return true;
  if (gsfIter->isEE() && isEB_)
    return true;
  if (gsfIter->isEBEEGap() && isNotEBEEGap_)
    return true;

  if (gsfIter->ecalDrivenSeed() && isTrackerDriven_)
    return true;
  if (gsfIter->trackerDrivenSeed() && isEcalDriven_)
    return true;

  return false;
}

bool ElectronTagProbeAnalyzer::etCut(const reco::GsfElectronCollection::const_iterator& gsfIter) {
  if (gsfIter->superCluster()->energy() / cosh(gsfIter->superCluster()->eta()) < minEt_)
    return true;

  return false;
}

bool ElectronTagProbeAnalyzer::isolationCut(const reco::GsfElectronCollection::const_iterator& gsfIter,
                                            double vertexTIP) {
  if (gsfIter->isEB() && vertexTIP > tipMaxBarrel_)
    return true;
  if (gsfIter->isEE() && vertexTIP > tipMaxEndcaps_)
    return true;

  if (gsfIter->dr03TkSumPt() > tkIso03Max_)
    return true;
  if (gsfIter->isEB() && gsfIter->dr03HcalTowerSumEt(1) > hcalIso03Depth1MaxBarrel_)
    return true;
  if (gsfIter->isEE() && gsfIter->dr03HcalTowerSumEt(1) > hcalIso03Depth1MaxEndcaps_)
    return true;
  if (gsfIter->isEE() && gsfIter->dr03HcalTowerSumEt(2) > hcalIso03Depth2MaxEndcaps_)
    return true;
  if (gsfIter->isEB() && gsfIter->dr03EcalRecHitSumEt() > ecalIso03MaxBarrel_)
    return true;
  if (gsfIter->isEE() && gsfIter->dr03EcalRecHitSumEt() > ecalIso03MaxEndcaps_)
    return true;

  return false;
}

bool ElectronTagProbeAnalyzer::idCut(const reco::GsfElectronCollection::const_iterator& gsfIter) {
  if (gsfIter->isEB() && gsfIter->eSuperClusterOverP() < eOverPMinBarrel_)
    return true;
  if (gsfIter->isEB() && gsfIter->eSuperClusterOverP() > eOverPMaxBarrel_)
    return true;
  if (gsfIter->isEE() && gsfIter->eSuperClusterOverP() < eOverPMinEndcaps_)
    return true;
  if (gsfIter->isEE() && gsfIter->eSuperClusterOverP() > eOverPMaxEndcaps_)
    return true;
  if (gsfIter->isEB() && std::abs(gsfIter->deltaEtaSuperClusterTrackAtVtx()) < dEtaMinBarrel_)
    return true;
  if (gsfIter->isEB() && std::abs(gsfIter->deltaEtaSuperClusterTrackAtVtx()) > dEtaMaxBarrel_)
    return true;
  if (gsfIter->isEE() && std::abs(gsfIter->deltaEtaSuperClusterTrackAtVtx()) < dEtaMinEndcaps_)
    return true;
  if (gsfIter->isEE() && std::abs(gsfIter->deltaEtaSuperClusterTrackAtVtx()) > dEtaMaxEndcaps_)
    return true;
  if (gsfIter->isEB() && std::abs(gsfIter->deltaPhiSuperClusterTrackAtVtx()) < dPhiMinBarrel_)
    return true;
  if (gsfIter->isEB() && std::abs(gsfIter->deltaPhiSuperClusterTrackAtVtx()) > dPhiMaxBarrel_)
    return true;
  if (gsfIter->isEE() && std::abs(gsfIter->deltaPhiSuperClusterTrackAtVtx()) < dPhiMinEndcaps_)
    return true;
  if (gsfIter->isEE() && std::abs(gsfIter->deltaPhiSuperClusterTrackAtVtx()) > dPhiMaxEndcaps_)
    return true;
  if (gsfIter->isEB() && gsfIter->scSigmaIEtaIEta() < sigIetaIetaMinBarrel_)
    return true;
  if (gsfIter->isEB() && gsfIter->scSigmaIEtaIEta() > sigIetaIetaMaxBarrel_)
    return true;
  if (gsfIter->isEE() && gsfIter->scSigmaIEtaIEta() < sigIetaIetaMinEndcaps_)
    return true;
  if (gsfIter->isEE() && gsfIter->scSigmaIEtaIEta() > sigIetaIetaMaxEndcaps_)
    return true;
  if (gsfIter->isEB() && gsfIter->hadronicOverEm() > hadronicOverEmMaxBarrel_)
    return true;
  if (gsfIter->isEE() && gsfIter->hadronicOverEm() > hadronicOverEmMaxEndcaps_)
    return true;

  return false;
}
