#include "DQMOffline/EGamma/plugins/ElectronAnalyzer.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
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

using namespace reco ;

ElectronAnalyzer::ElectronAnalyzer( const edm::ParameterSet & conf )
 : ElectronDqmAnalyzerBase(conf)
 {
  // general, collections
  Selection_ = conf.getParameter<int>("Selection");
  electronCollection_ = consumes<GsfElectronCollection>(conf.getParameter<edm::InputTag>("ElectronCollection"));
  matchingObjectCollection_ = consumes<SuperClusterCollection>(conf.getParameter<edm::InputTag>("MatchingObjectCollection"));
  trackCollection_ = consumes<TrackCollection>(conf.getParameter<edm::InputTag>("TrackCollection"));
  vertexCollection_ = consumes<VertexCollection>(conf.getParameter<edm::InputTag>("VertexCollection"));
  gsftrackCollection_ = consumes<GsfTrackCollection>(conf.getParameter<edm::InputTag>("GsfTrackCollection"));
  beamSpotTag_ = consumes<BeamSpot>(conf.getParameter<edm::InputTag>("BeamSpot"));
  readAOD_ = conf.getParameter<bool>("ReadAOD");

  // matching
  matchingCondition_ = conf.getParameter<std::string>("MatchingCondition");
  assert (matchingCondition_=="Cone") ;
  maxPtMatchingObject_ = conf.getParameter<double>("MaxPtMatchingObject");
  maxAbsEtaMatchingObject_ = conf.getParameter<double>("MaxAbsEtaMatchingObject");
  deltaR_ = conf.getParameter<double>("DeltaR");

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

  // histos limits and binning
//  edm::ParameterSet histosSet = conf.getParameter<edm::ParameterSet>("histosCfg") ;

  nbineta=conf.getParameter<int>("NbinEta");
  nbineta2D=conf.getParameter<int>("NbinEta2D");
  etamin=conf.getParameter<double>("EtaMin");
  etamax=conf.getParameter<double>("EtaMax");
  //
  nbinphi=conf.getParameter<int>("NbinPhi");
  nbinphi2D=conf.getParameter<int>("NbinPhi2D");
  phimin=conf.getParameter<double>("PhiMin");
  phimax=conf.getParameter<double>("PhiMax");
  //
  nbinpt=conf.getParameter<int>("NbinPt");
  nbinpteff=conf.getParameter<int>("NbinPtEff");
  nbinpt2D=conf.getParameter<int>("NbinPt2D");
  ptmax=conf.getParameter<double>("PtMax");
  //
  nbinp=conf.getParameter<int>("NbinP");
  nbinp2D=conf.getParameter<int>("NbinP2D");
  pmax=conf.getParameter<double>("PMax");
  //
  nbineop=conf.getParameter<int>("NbinEop");
  nbineop2D=conf.getParameter<int>("NbinEop2D");
  eopmax=conf.getParameter<double>("EopMax");
  eopmaxsht=conf.getParameter<double>("EopMaxSht");
  //
  nbindeta=conf.getParameter<int>("NbinDeta");
  detamin=conf.getParameter<double>("DetaMin");
  detamax=conf.getParameter<double>("DetaMax");
  //
  nbindphi=conf.getParameter<int>("NbinDphi");
  dphimin=conf.getParameter<double>("DphiMin");
  dphimax=conf.getParameter<double>("DphiMax");
  //
  nbindetamatch=conf.getParameter<int>("NbinDetaMatch");
  nbindetamatch2D=conf.getParameter<int>("NbinDetaMatch2D");
  detamatchmin=conf.getParameter<double>("DetaMatchMin");
  detamatchmax=conf.getParameter<double>("DetaMatchMax");
  //
  nbindphimatch=conf.getParameter<int>("NbinDphiMatch");
  nbindphimatch2D=conf.getParameter<int>("NbinDphiMatch2D");
  dphimatchmin=conf.getParameter<double>("DphiMatchMin");
  dphimatchmax=conf.getParameter<double>("DphiMatchMax");
  //
  nbinfhits=conf.getParameter<int>("NbinFhits");
  fhitsmax=conf.getParameter<double>("FhitsMax");
  //
  nbinlhits=conf.getParameter<int>("NbinLhits");
  lhitsmax=conf.getParameter<double>("LhitsMax");
  //
  nbinxyz=conf.getParameter<int>("NbinXyz");
  nbinxyz2D=conf.getParameter<int>("NbinXyz2D");
  //
  nbinpoptrue= conf.getParameter<int>("NbinPopTrue");
  poptruemin=conf.getParameter<double>("PopTrueMin");
  poptruemax=conf.getParameter<double>("PopTrueMax");
  //
  nbinmee= conf.getParameter<int>("NbinMee");
  meemin=conf.getParameter<double>("MeeMin");
  meemax=conf.getParameter<double>("MeeMax");
  //
  nbinhoe= conf.getParameter<int>("NbinHoe");
  hoemin=conf.getParameter<double>("HoeMin");
  hoemax=conf.getParameter<double>("HoeMax");

//  set_EfficiencyFlag=histosSet.getParameter<bool>("EfficiencyFlag");
//  set_StatOverflowFlag=histosSet.getParameter<bool>("StatOverflowFlag");
 }

ElectronAnalyzer::~ElectronAnalyzer()
 {}

void ElectronAnalyzer::bookHistograms( DQMStore::IBooker & iBooker, edm::Run const &, edm::EventSetup const & )
 {
  iBooker.setCurrentFolder(outputInternalPath_) ;

  nEvents_ = 0 ;

  // basic quantities
  h1_vertexPt_barrel = bookH1(iBooker, "vertexPt_barrel","ele transverse momentum in barrel",nbinpt,0.,ptmax,"p_{T vertex} (GeV/c)");
  h1_vertexPt_endcaps = bookH1(iBooker, "vertexPt_endcaps","ele transverse momentum in endcaps",nbinpt,0.,ptmax,"p_{T vertex} (GeV/c)");
  h1_vertexEta = bookH1(iBooker, "vertexEta","ele momentum #eta",nbineta,etamin,etamax,"#eta");
  h2_vertexEtaVsPhi = bookH2(iBooker, "vertexEtaVsPhi","ele momentum #eta vs #phi",nbineta2D,etamin,etamax,nbinphi2D,phimin,phimax,"#eta","#phi (rad)");
  h2_vertexXvsY = bookH2(iBooker, "vertexXvsY","ele vertex x vs y",nbinxyz2D,-0.1,0.1,nbinxyz2D,-0.1,0.1,"x (cm)","y (cm)");
  h1_vertexZ = bookH1(iBooker, "vertexZ","ele vertex z",nbinxyz,-25, 25,"z (cm)");

  // super-clusters
  h1_sclEt = bookH1(iBooker, "sclEt","ele supercluster transverse energy",nbinpt,0.,ptmax,"E_{T} (GeV)");

  // electron track
  h1_chi2 = bookH1(iBooker, "chi2","ele track #chi^{2}",100,0.,15.,"#Chi^{2}");
  py_chi2VsEta = bookP1(iBooker, "chi2VsEta","ele track #chi^{2} vs #eta",nbineta2D,etamin,etamax,0.,15.,"#eta","<#chi^{2}>");
  py_chi2VsPhi = bookP1(iBooker, "chi2VsPhi","ele track #chi^{2} vs #phi",nbinphi2D,phimin,phimax,0.,15.,"#phi (rad)","<#chi^{2}>");
  h1_foundHits = bookH1(iBooker, "foundHits","ele track # found hits",nbinfhits,0.,fhitsmax,"N_{hits}");
  py_foundHitsVsEta = bookP1(iBooker, "foundHitsVsEta","ele track # found hits vs #eta",nbineta2D,etamin,etamax,0.,fhitsmax,"#eta","<# hits>");
  py_foundHitsVsPhi = bookP1(iBooker, "foundHitsVsPhi","ele track # found hits vs #phi",nbinphi2D,phimin,phimax,0.,fhitsmax,"#phi (rad)","<# hits>");
  h1_lostHits = bookH1(iBooker, "lostHits","ele track # lost hits",5,0.,5.,"N_{lost hits}");
  py_lostHitsVsEta = bookP1(iBooker, "lostHitsVsEta","ele track # lost hits vs #eta",nbineta2D,etamin,etamax,0.,lhitsmax,"#eta","<# hits>");
  py_lostHitsVsPhi = bookP1(iBooker, "lostHitsVsPhi","ele track # lost hits vs #eta",nbinphi2D,phimin,phimax,0.,lhitsmax,"#phi (rad)","<# hits>");

  // electron matching and ID
  h1_Eop_barrel = bookH1(iBooker, "Eop_barrel","ele E/P_{vertex} in barrel",nbineop,0.,eopmax,"E/P_{vertex}");
  h1_Eop_endcaps = bookH1(iBooker, "Eop_endcaps","ele E/P_{vertex} in endcaps",nbineop,0.,eopmax,"E/P_{vertex}");
  py_EopVsPhi = bookP1(iBooker, "EopVsPhi","ele E/P_{vertex} vs #phi",nbinphi2D,phimin,phimax,0.,eopmax,"#phi (rad)","<E/P_{vertex}>");
  h1_EeleOPout_barrel = bookH1(iBooker, "EeleOPout_barrel","ele E_{ele}/P_{out} in barrel",nbineop,0.,eopmax,"E_{ele}/P_{out}");
  h1_EeleOPout_endcaps = bookH1(iBooker, "EeleOPout_endcaps","ele E_{ele}/P_{out} in endcaps",nbineop,0.,eopmax,"E_{ele}/P_{out}");
  h1_dEtaSc_propVtx_barrel = bookH1(iBooker, "dEtaSc_propVtx_barrel","ele #eta_{sc} - #eta_{tr}, prop from vertex, in barrel",nbindetamatch,detamatchmin,detamatchmax,"#eta_{sc} - #eta_{tr}");
  h1_dEtaSc_propVtx_endcapsPos = bookH1(iBooker, "dEtaSc_propVtx_endcapsPos","ele #eta_{sc} - #eta_{tr}, prop from vertex, in positive endcap",nbindetamatch,detamatchmin,detamatchmax,"#eta_{sc} - #eta_{tr}");
  h1_dEtaSc_propVtx_endcapsNeg = bookH1(iBooker, "dEtaSc_propVtx_endcapsNeg","ele #eta_{sc} - #eta_{tr}, prop from vertex, in negative endcap",nbindetamatch,detamatchmin,detamatchmax,"#eta_{sc} - #eta_{tr}");
  py_dEtaSc_propVtxVsPhi = bookP1(iBooker, "dEtaSc_propVtxVsPhi","ele #eta_{sc} - #eta_{tr}, prop from vertex vs #phi",nbinphi2D,phimin,phimax,detamatchmin,detamatchmax,"#phi (rad)","<#eta_{sc} - #eta_{tr}>");
  h1_dEtaEleCl_propOut_barrel = bookH1(iBooker, "dEtaEleCl_propOut_barrel","ele #eta_{EleCl} - #eta_{tr}, prop from outermost, in barrel",nbindetamatch,detamatchmin,detamatchmax,"#eta_{elecl} - #eta_{tr}");
  h1_dEtaEleCl_propOut_endcapsPos = bookH1(iBooker, "dEtaEleCl_propOut_endcapsPos","ele #eta_{EleCl} - #eta_{tr}, prop from outermost, in positive endcap",nbindetamatch,detamatchmin,detamatchmax,"#eta_{elecl} - #eta_{tr}");
  h1_dEtaEleCl_propOut_endcapsNeg = bookH1(iBooker, "dEtaEleCl_propOut_endcapsNeg","ele #eta_{EleCl} - #eta_{tr}, prop from outermost, in negative endcap",nbindetamatch,detamatchmin,detamatchmax,"#eta_{elecl} - #eta_{tr}");
  h1_dPhiSc_propVtx_barrel = bookH1(iBooker, "dPhiSc_propVtx_barrel","ele #phi_{sc} - #phi_{tr}, prop from vertex, in barrel",nbindphimatch,dphimatchmin,dphimatchmax,"#phi_{sc} - #phi_{tr} (rad)");
  h1_dPhiSc_propVtx_endcapsPos = bookH1(iBooker, "dPhiSc_propVtx_endcapsPos","ele #phi_{sc} - #phi_{tr}, prop from vertex, in positive endcap",nbindphimatch,dphimatchmin,dphimatchmax,"#phi_{sc} - #phi_{tr} (rad)");
  h1_dPhiSc_propVtx_endcapsNeg = bookH1(iBooker, "dPhiSc_propVtx_endcapsNeg","ele #phi_{sc} - #phi_{tr}, prop from vertex, in negative endcap",nbindphimatch,dphimatchmin,dphimatchmax,"#phi_{sc} - #phi_{tr} (rad)");
  py_dPhiSc_propVtxVsPhi = bookP1(iBooker, "dPhiSc_propVtxVsPhi","ele #phi_{sc} - #phi_{tr}, prop from vertex vs #phi",nbinphi2D,phimin,phimax,dphimatchmin,dphimatchmax,"#phi (rad)","<#phi_{sc} - #phi_{tr}> (rad)");
  h1_dPhiEleCl_propOut_barrel = bookH1(iBooker, "dPhiEleCl_propOut_barrel","ele #phi_{EleCl} - #phi_{tr}, prop from outermost, in barrel",nbindphimatch,dphimatchmin,dphimatchmax,"#phi_{elecl} - #phi_{tr} (rad)");
  h1_dPhiEleCl_propOut_endcapsPos = bookH1(iBooker, "dPhiEleCl_propOut_endcapsPos","ele #phi_{EleCl} - #phi_{tr}, prop from outermost, in positive endcap",nbindphimatch,dphimatchmin,dphimatchmax,"#phi_{elecl} - #phi_{tr} (rad)");
  h1_dPhiEleCl_propOut_endcapsNeg = bookH1(iBooker, "dPhiEleCl_propOut_endcapsNeg","ele #phi_{EleCl} - #phi_{tr}, prop from outermost, in negative endcap",nbindphimatch,dphimatchmin,dphimatchmax,"#phi_{elecl} - #phi_{tr} (rad)");
  h1_Hoe_barrel = bookH1(iBooker, "Hoe_barrel","ele hadronic energy / em energy, in barrel", nbinhoe, hoemin, hoemax,"H/E","Events","ELE_LOGY E1 P") ;
  h1_Hoe_endcaps = bookH1(iBooker, "Hoe_endcaps","ele hadronic energy / em energy, in endcaps", nbinhoe, hoemin, hoemax,"H/E","Events","ELE_LOGY E1 P") ;
  py_HoeVsPhi = bookP1(iBooker, "HoeVsPhi","ele hadronic energy / em energy vs #phi",nbinphi2D,phimin,phimax,hoemin,hoemax,"#phi (rad)","<H/E>","E1 P") ;
  h1_sclSigEtaEta_barrel = bookH1(iBooker, "sclSigEtaEta_barrel","ele sigma eta eta in barrel",100,0.,0.05,"sietaieta");
  h1_sclSigEtaEta_endcaps = bookH1(iBooker, "sclSigEtaEta_endcaps","ele sigma eta eta in endcaps",100,0.,0.05,"sietaieta");
  h1_sigIEtaIEta5x5_barrel = bookH1(iBooker, "sigIEtaIEta5x5_barrel","ele sigma ieta ieta 5x5 in barrel",100,0.,0.05,"sietaieta5x5");
  h1_sigIEtaIEta5x5_endcaps = bookH1(iBooker, "sigIEtaIEta5x5_endcaps","ele sigma ieta ieta 5x5 in endcaps",100,0.,0.05,"sietaieta5x5");

  // fbrem
  h1_fbrem = bookH1(iBooker, "fbrem","ele brem fraction",100,0.,1.,"P_{in} - P_{out} / P_{in}") ;
  py_fbremVsEta = bookP1(iBooker, "fbremVsEta","ele brem fraction vs #eta",nbineta2D,etamin,etamax,0.,1.,"#eta","<P_{in} - P_{out} / P_{in}>") ;
  py_fbremVsPhi = bookP1(iBooker, "fbremVsPhi","ele brem fraction vs #phi",nbinphi2D,phimin,phimax,0.,1.,"#phi (rad)","<P_{in} - P_{out} / P_{in}>") ;
  h1_classes = bookH1(iBooker, "classes","ele electron classes",10,0.0,10.);

  // pflow
  h1_mva = bookH1(iBooker, "mva","ele identification mva",100,-1.,1.,"mva");
  h1_provenance = bookH1(iBooker, "provenance","ele provenance",5,-2.,3.,"provenance");

  // isolation
  h1_tkSumPt_dr03 = bookH1(iBooker, "tkSumPt_dr03","tk isolation sum, dR=0.3",100,0.0,20.,"TkIsoSum (GeV/c)","Events","ELE_LOGY E1 P");
  h1_ecalRecHitSumEt_dr03 = bookH1(iBooker, "ecalRecHitSumEt_dr03","ecal isolation sum, dR=0.3",100,0.0,20.,"EcalIsoSum (GeV)","Events","ELE_LOGY E1 P");
  h1_hcalTowerSumEt_dr03 = bookH1(iBooker, "hcalTowerSumEt_dr03","hcal isolation sum, dR=0.3",100,0.0,20.,"HcalIsoSum (GeV)","Events","ELE_LOGY E1 P");

  // pf isolation
  h1_PFch_dr03 = bookH1(iBooker, "PFch_dr03","Charged PF candidate sum, dR=0.3",100,0.0,20.,"PF charged (GeV/c)","Events","ELE_LOGY E1 P");
  h1_PFem_dr03 = bookH1(iBooker, "PFem_dr03","Neutral EM PF candidate sum, dR=0.3",100,0.0,20.,"PF neutral EM (GeV)","Events","ELE_LOGY E1 P");
  h1_PFnh_dr03 = bookH1(iBooker, "PFnh_dr03","Neutral Had PF candidate sum, dR=0.3",100,0.0,20.,"PF neutral Had (GeV)","Events","ELE_LOGY E1 P");

  // di-electron mass
  setBookIndex(200) ;
  h1_mee = bookH1(iBooker, "mee","ele pairs invariant mass", nbinmee, meemin, meemax,"m_{ee} (GeV/c^{2})");
  h1_mee_os = bookH1(iBooker, "mee_os","ele pairs invariant mass, opposite sign", nbinmee, meemin, meemax,"m_{e^{+}e^{-}} (GeV/c^{2})");

  //===========================
  // histos for matching and matched matched objects
  //===========================

  // matching object
  std::string matchingObjectType ;
  Labels l;
  labelsForToken(matchingObjectCollection_,l);
  if (std::string::npos != std::string(l.module).find("SuperCluster",0))
    { matchingObjectType = "SC" ; }
  if (matchingObjectType=="")
    { edm::LogError("ElectronMcFakeValidator::beginJob")<<"Unknown matching object type !" ; }
  else
   { edm::LogInfo("ElectronMcFakeValidator::beginJob")<<"Matching object type: "<<matchingObjectType ; }

  // matching object distributions
  h1_matchingObject_Eta = bookH1withSumw2(iBooker, "matchingObject_Eta",matchingObjectType+" #eta",nbineta,etamin,etamax,"#eta_{SC}");


  h1_matchingObject_Pt = bookH1withSumw2(iBooker, "matchingObject_Pt",matchingObjectType+" pt",nbinpteff,5.,ptmax,"pt_{SC} (GeV/c)");
  h1_matchingObject_Phi = bookH1withSumw2(iBooker, "matchingObject_Phi",matchingObjectType+" #phi",nbinphi,phimin,phimax,"#phi (rad)");

  h1_matchedObject_Eta = bookH1withSumw2(iBooker, "matchedObject_Eta","Efficiency vs matching SC #eta",nbineta,etamin,etamax,"#eta_{SC}");
  h1_matchedObject_Pt = bookH1withSumw2(iBooker, "matchedObject_Pt","Efficiency vs matching SC E_{T}",nbinpteff,5.,ptmax,"pt_{SC} (GeV/c)");
  h1_matchedObject_Phi = bookH1withSumw2(iBooker, "matchedObject_Phi","Efficiency vs matching SC #phi",nbinphi,phimin,phimax,"#phi (rad)");
  /**/

  }

void ElectronAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup & iSetup )
{
  nEvents_++ ;

  edm::Handle<GsfElectronCollection> gsfElectrons ;
  iEvent.getByToken(electronCollection_,gsfElectrons) ;
  edm::Handle<reco::SuperClusterCollection> recoClusters ;
  iEvent.getByToken(matchingObjectCollection_,recoClusters) ;
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByToken(trackCollection_,tracks);
  edm::Handle<reco::GsfTrackCollection> gsfTracks;
  iEvent.getByToken(gsftrackCollection_,gsfTracks);
  edm::Handle<reco::VertexCollection> vertices;
  iEvent.getByToken(vertexCollection_,vertices);
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle ;
  iEvent.getByToken(beamSpotTag_,recoBeamSpotHandle) ;
  const BeamSpot bs = *recoBeamSpotHandle ;

  edm::EventNumber_t ievt = iEvent.id().event();
  edm::RunNumber_t irun = iEvent.id().run();
  edm::LuminosityBlockNumber_t ils = iEvent.luminosityBlock();

  edm::LogInfo("ElectronAnalyzer::analyze")
    <<"Treating "<<gsfElectrons.product()->size()<<" electrons"
    <<" from event "<<ievt<<" in run "<<irun<<" and lumiblock "<<ils ;
  //h1_num_->Fill((*gsfElectrons).size()) ;

  // selected rec electrons
  reco::GsfElectronCollection::const_iterator gsfIter ;
  for
   ( gsfIter=gsfElectrons->begin() ;
     gsfIter!=gsfElectrons->end();
     gsfIter++ )
   {
    // vertex TIP
    double vertexTIP =
     (gsfIter->vertex().x()-bs.position().x()) * (gsfIter->vertex().x()-bs.position().x()) +
     (gsfIter->vertex().y()-bs.position().y()) * (gsfIter->vertex().y()-bs.position().y()) ;
    vertexTIP = sqrt(vertexTIP) ;

    // select electrons
    if (!selected(gsfIter,vertexTIP)) continue ;

    // invariant mass of 2 selected electrons
    reco::GsfElectronCollection::const_iterator gsfIter2 ;
    for
      ( gsfIter2=gsfIter+1;
	gsfIter2!=gsfElectrons->end() ;
	gsfIter2++ )
      {
        if (!selected(gsfIter2,vertexTIP)) continue ;
	float invMass = computeInvMass(*gsfIter,*gsfIter2) ;
        h1_mee->Fill(invMass) ;
        if ( ( (gsfIter->charge())*(gsfIter2->charge()) )<0. )
	  { h1_mee_os->Fill(invMass) ; }
      }
    
    // basic quantities
    if (gsfIter->isEB()) h1_vertexPt_barrel->Fill( gsfIter->pt() );
    if (gsfIter->isEE()) h1_vertexPt_endcaps->Fill( gsfIter->pt() );
    h1_vertexEta->Fill( gsfIter->eta() );
    h2_vertexEtaVsPhi->Fill( gsfIter->eta(), gsfIter->phi() );
    h2_vertexXvsY->Fill( gsfIter->vertex().x(), gsfIter->vertex().y() );
    h1_vertexZ->Fill( gsfIter->vertex().z() );

    // supercluster related distributions
    reco::SuperClusterRef sclRef = gsfIter->superCluster() ;
    // ALREADY DONE IN GSF ELECTRON CORE
    double R=TMath::Sqrt(sclRef->x()*sclRef->x() + sclRef->y()*sclRef->y() +sclRef->z()*sclRef->z());
    double Rt=TMath::Sqrt(sclRef->x()*sclRef->x() + sclRef->y()*sclRef->y());
    h1_sclEt->Fill(sclRef->energy()*(Rt/R));

    // track related distributions
    if (!readAOD_)
     { // track extra does not exist in AOD
      h1_foundHits->Fill( gsfIter->gsfTrack()->numberOfValidHits() );
      py_foundHitsVsEta->Fill( gsfIter->eta(), gsfIter->gsfTrack()->numberOfValidHits() );
      py_foundHitsVsPhi->Fill( gsfIter->phi(), gsfIter->gsfTrack()->numberOfValidHits() );
      h1_lostHits->Fill( gsfIter->gsfTrack()->numberOfLostHits() );
      py_lostHitsVsEta->Fill( gsfIter->eta(), gsfIter->gsfTrack()->numberOfLostHits() );
      py_lostHitsVsPhi->Fill( gsfIter->phi(), gsfIter->gsfTrack()->numberOfLostHits() );
      h1_chi2->Fill( gsfIter->gsfTrack()->normalizedChi2() );
      py_chi2VsEta->Fill( gsfIter->eta(), gsfIter->gsfTrack()->normalizedChi2() );
      py_chi2VsPhi->Fill( gsfIter->phi(), gsfIter->gsfTrack()->normalizedChi2() );
     }

    // match distributions
    if (gsfIter->isEB())
     {
      h1_Eop_barrel->Fill( gsfIter->eSuperClusterOverP() );
      h1_EeleOPout_barrel->Fill( gsfIter->eEleClusterOverPout() );
      h1_dEtaSc_propVtx_barrel->Fill(gsfIter->deltaEtaSuperClusterTrackAtVtx());
      h1_dEtaEleCl_propOut_barrel->Fill(gsfIter->deltaEtaEleClusterTrackAtCalo());
      h1_dPhiSc_propVtx_barrel->Fill(gsfIter->deltaPhiSuperClusterTrackAtVtx());
      h1_dPhiEleCl_propOut_barrel->Fill(gsfIter->deltaPhiEleClusterTrackAtCalo());
      h1_Hoe_barrel->Fill(gsfIter->hadronicOverEm());
      h1_sclSigEtaEta_barrel->Fill( gsfIter->scSigmaEtaEta() );
      h1_sigIEtaIEta5x5_barrel->Fill( gsfIter->full5x5_sigmaIetaIeta() );
     }
    if (gsfIter->isEE())
     {
      h1_Eop_endcaps->Fill( gsfIter->eSuperClusterOverP() );
      h1_EeleOPout_endcaps->Fill( gsfIter->eEleClusterOverPout() );
      if (gsfIter->eta() > 0) {
	h1_dEtaSc_propVtx_endcapsPos->Fill(gsfIter->deltaEtaSuperClusterTrackAtVtx());
	h1_dEtaEleCl_propOut_endcapsPos->Fill(gsfIter->deltaEtaEleClusterTrackAtCalo());
	h1_dPhiSc_propVtx_endcapsPos->Fill(gsfIter->deltaPhiSuperClusterTrackAtVtx());
	h1_dPhiEleCl_propOut_endcapsPos->Fill(gsfIter->deltaPhiEleClusterTrackAtCalo());
      }
      else {
	h1_dEtaSc_propVtx_endcapsNeg->Fill(gsfIter->deltaEtaSuperClusterTrackAtVtx());
	h1_dEtaEleCl_propOut_endcapsNeg->Fill(gsfIter->deltaEtaEleClusterTrackAtCalo());
	h1_dPhiSc_propVtx_endcapsNeg->Fill(gsfIter->deltaPhiSuperClusterTrackAtVtx());
	h1_dPhiEleCl_propOut_endcapsNeg->Fill(gsfIter->deltaPhiEleClusterTrackAtCalo());
      }
      h1_Hoe_endcaps->Fill(gsfIter->hadronicOverEm());
      h1_sclSigEtaEta_endcaps->Fill( gsfIter->scSigmaEtaEta() );
      h1_sigIEtaIEta5x5_endcaps->Fill( gsfIter->full5x5_sigmaIetaIeta() );
     }
    py_EopVsPhi->Fill( gsfIter->phi(), gsfIter->eSuperClusterOverP() );
    py_dEtaSc_propVtxVsPhi->Fill(gsfIter->phi(), gsfIter->deltaEtaSuperClusterTrackAtVtx());
    py_dPhiSc_propVtxVsPhi->Fill(gsfIter->phi(), gsfIter->deltaPhiSuperClusterTrackAtVtx());
    py_HoeVsPhi->Fill(gsfIter->phi(), gsfIter->hadronicOverEm());

    // fbrem, classes
    h1_fbrem->Fill(gsfIter->fbrem()) ;
    py_fbremVsEta->Fill(gsfIter->eta(),gsfIter->fbrem()) ;
    py_fbremVsPhi->Fill(gsfIter->phi(),gsfIter->fbrem()) ;
    int eleClass = gsfIter->classification() ;
    if (gsfIter->isEE()) eleClass+=5;
    h1_classes->Fill(eleClass) ;


    // pflow
    h1_mva->Fill(gsfIter->mva_e_pi()) ;
    if (gsfIter->ecalDrivenSeed()) h1_provenance->Fill(1.) ;
    if (gsfIter->trackerDrivenSeed()) h1_provenance->Fill(-1.) ;
    if (gsfIter->trackerDrivenSeed()||gsfIter->ecalDrivenSeed()) h1_provenance->Fill(0.);
    if (gsfIter->trackerDrivenSeed()&&!gsfIter->ecalDrivenSeed()) h1_provenance->Fill(-2.);
    if (!gsfIter->trackerDrivenSeed()&&gsfIter->ecalDrivenSeed()) h1_provenance->Fill(2.);

    // isolation
    h1_tkSumPt_dr03->Fill(gsfIter->dr03TkSumPt());
    h1_ecalRecHitSumEt_dr03->Fill(gsfIter->dr03EcalRecHitSumEt());
    h1_hcalTowerSumEt_dr03->Fill(gsfIter->dr03HcalTowerSumEt());

    // PF isolation
    GsfElectron::PflowIsolationVariables pfIso = gsfIter->pfIsolationVariables();
    h1_PFch_dr03->Fill( pfIso.sumChargedHadronPt ); 
    h1_PFem_dr03->Fill( pfIso.sumPhotonEt ); 
    h1_PFnh_dr03->Fill( pfIso.sumNeutralHadronEt ); 


   }

  // association matching object-reco electrons
  int matchingObjectNum=0;
  reco::SuperClusterCollection::const_iterator moIter ;
  for
   ( moIter=recoClusters->begin() ;
     moIter!=recoClusters->end() ;
     moIter++ )
   {
//    // number of matching objects
     matchingObjectNum++;

    if
     ( moIter->energy()/cosh(moIter->eta())>maxPtMatchingObject_ ||
       std::abs(moIter->eta())> maxAbsEtaMatchingObject_ )
     { continue ; }

//    // suppress the endcaps
    h1_matchingObject_Eta->Fill( moIter->eta() );
    h1_matchingObject_Pt->Fill( moIter->energy()/cosh(moIter->eta()) );
    h1_matchingObject_Phi->Fill( moIter->phi() );

    bool okGsfFound = false ;
    double gsfOkRatio = 999999999. ;
    reco::GsfElectron bestGsfElectron ;
    reco::GsfElectronCollection::const_iterator gsfIter ;
    for
     ( gsfIter=gsfElectrons->begin() ;
       gsfIter!=gsfElectrons->end() ;
       gsfIter++ )
     {

      double vertexTIP =
       (gsfIter->vertex().x()-bs.position().x()) * (gsfIter->vertex().x()-bs.position().x()) +
       (gsfIter->vertex().y()-bs.position().y()) * (gsfIter->vertex().y()-bs.position().y()) ;
      vertexTIP = sqrt(vertexTIP) ;

      // select electrons
      if (!selected(gsfIter,vertexTIP)) continue ;

      // matching with a cone in eta phi
      if ( matchingCondition_ == "Cone" )
       {
        double dphi = gsfIter->phi()-moIter->phi() ;
        if (std::abs(dphi)>CLHEP::pi)
         { dphi = dphi < 0? (CLHEP::twopi) + dphi : dphi - CLHEP::twopi ; }
        double deltaR = sqrt(pow((moIter->eta()-gsfIter->eta()),2) + pow(dphi,2)) ;
        if ( deltaR < deltaR_ )
         {
          double tmpGsfRatio = gsfIter->p()/moIter->energy() ;
          if ( std::abs(tmpGsfRatio-1) < std::abs(gsfOkRatio-1) )
           {
            gsfOkRatio = tmpGsfRatio;
            bestGsfElectron=*gsfIter;
            okGsfFound = true;
           }
         }
       }
     } // loop over rec ele to look for the best one
    if (okGsfFound)
     {
      // generated distributions for matched electrons
      h1_matchedObject_Eta->Fill( moIter->eta() );
      h1_matchedObject_Pt->Fill( moIter->energy()/cosh(moIter->eta()) );
      h1_matchedObject_Phi->Fill( moIter->phi() );

      //classes
     }

   } // loop overmatching object


 }

float ElectronAnalyzer::computeInvMass
 ( const reco::GsfElectron & e1,
   const reco::GsfElectron & e2 )
 {
  math::XYZTLorentzVector p12 = e1.p4()+e2.p4() ;
  float mee2 = p12.Dot(p12) ;
  float invMass = mee2 > 0. ? sqrt(mee2) : 0;
  return invMass ;
 }

bool ElectronAnalyzer::selected( const reco::GsfElectronCollection::const_iterator & gsfIter , double vertexTIP )
 {
  if ((Selection_>0)&&generalCut(gsfIter)) return false ;
  if ((Selection_>=1)&&etCut(gsfIter)) return false ;
  if ((Selection_>=2)&&isolationCut(gsfIter,vertexTIP)) return false ;
  if ((Selection_>=3)&&idCut(gsfIter)) return false ;
  return true ;
 }

bool ElectronAnalyzer::generalCut( const reco::GsfElectronCollection::const_iterator & gsfIter)
 {
  if (std::abs(gsfIter->eta())>maxAbsEta_) return true ;
  if (gsfIter->pt()<minPt_) return true ;

  if (gsfIter->isEB() && isEE_) return true ;
  if (gsfIter->isEE() && isEB_) return true ;
  if (gsfIter->isEBEEGap() && isNotEBEEGap_) return true ;

  if (gsfIter->ecalDrivenSeed() && isTrackerDriven_) return true ;
  if (gsfIter->trackerDrivenSeed() && isEcalDriven_) return true ;

  return false ;
 }

bool ElectronAnalyzer::etCut( const reco::GsfElectronCollection::const_iterator & gsfIter )
 {
  if (gsfIter->superCluster()->energy()/cosh(gsfIter->superCluster()->eta())<minEt_) return true ;

  return false ;
 }

bool ElectronAnalyzer::isolationCut( const reco::GsfElectronCollection::const_iterator & gsfIter, double vertexTIP )
 {
  if (gsfIter->isEB() && vertexTIP > tipMaxBarrel_) return true ;
  if (gsfIter->isEE() && vertexTIP > tipMaxEndcaps_) return true ;

  if (gsfIter->dr03TkSumPt() > tkIso03Max_) return true ;
  if (gsfIter->isEB() && gsfIter->dr03HcalDepth1TowerSumEt() > hcalIso03Depth1MaxBarrel_) return true ;
  if (gsfIter->isEE() && gsfIter->dr03HcalDepth1TowerSumEt() > hcalIso03Depth1MaxEndcaps_) return true ;
  if (gsfIter->isEE() && gsfIter->dr03HcalDepth2TowerSumEt() > hcalIso03Depth2MaxEndcaps_) return true ;
  if (gsfIter->isEB() && gsfIter->dr03EcalRecHitSumEt() > ecalIso03MaxBarrel_) return true ;
  if (gsfIter->isEE() && gsfIter->dr03EcalRecHitSumEt() > ecalIso03MaxEndcaps_) return true ;

  return false ;
 }

bool ElectronAnalyzer::idCut( const reco::GsfElectronCollection::const_iterator & gsfIter )
 {
  if (gsfIter->isEB() && gsfIter->eSuperClusterOverP() < eOverPMinBarrel_) return true ;
  if (gsfIter->isEB() && gsfIter->eSuperClusterOverP() > eOverPMaxBarrel_) return true ;
  if (gsfIter->isEE() && gsfIter->eSuperClusterOverP() < eOverPMinEndcaps_) return true ;
  if (gsfIter->isEE() && gsfIter->eSuperClusterOverP() > eOverPMaxEndcaps_) return true ;
  if (gsfIter->isEB() && std::abs(gsfIter->deltaEtaSuperClusterTrackAtVtx()) < dEtaMinBarrel_) return true ;
  if (gsfIter->isEB() && std::abs(gsfIter->deltaEtaSuperClusterTrackAtVtx()) > dEtaMaxBarrel_) return true ;
  if (gsfIter->isEE() && std::abs(gsfIter->deltaEtaSuperClusterTrackAtVtx()) < dEtaMinEndcaps_) return true ;
  if (gsfIter->isEE() && std::abs(gsfIter->deltaEtaSuperClusterTrackAtVtx()) > dEtaMaxEndcaps_) return true ;
  if (gsfIter->isEB() && std::abs(gsfIter->deltaPhiSuperClusterTrackAtVtx()) < dPhiMinBarrel_) return true ;
  if (gsfIter->isEB() && std::abs(gsfIter->deltaPhiSuperClusterTrackAtVtx()) > dPhiMaxBarrel_) return true ;
  if (gsfIter->isEE() && std::abs(gsfIter->deltaPhiSuperClusterTrackAtVtx()) < dPhiMinEndcaps_) return true ;
  if (gsfIter->isEE() && std::abs(gsfIter->deltaPhiSuperClusterTrackAtVtx()) > dPhiMaxEndcaps_) return true ;
  if (gsfIter->isEB() && gsfIter->scSigmaIEtaIEta() < sigIetaIetaMinBarrel_) return true ;
  if (gsfIter->isEB() && gsfIter->scSigmaIEtaIEta() > sigIetaIetaMaxBarrel_) return true ;
  if (gsfIter->isEE() && gsfIter->scSigmaIEtaIEta() < sigIetaIetaMinEndcaps_) return true ;
  if (gsfIter->isEE() && gsfIter->scSigmaIEtaIEta() > sigIetaIetaMaxEndcaps_) return true ;
  if (gsfIter->isEB() && gsfIter->hadronicOverEm() > hadronicOverEmMaxBarrel_) return true ;
  if (gsfIter->isEE() && gsfIter->hadronicOverEm() > hadronicOverEmMaxEndcaps_) return true ;

  return false ;
 }
