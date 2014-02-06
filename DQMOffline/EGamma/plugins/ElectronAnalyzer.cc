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
//#include "FWCore/Framework/interface/EDAnalyzer.h"
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
//  HLTPathsByName_= conf.getParameter<std::vector<std::string > >("HltPaths");
//  HLTPathsByIndex_.resize(HLTPathsByName_.size());

  // histos limits and binning
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
 }

ElectronAnalyzer::~ElectronAnalyzer()
 {}

void ElectronAnalyzer::book()
 {
  nEvents_ = 0 ;
  //nAfterTrigger_ = 0 ;


  // basic quantities
//  h1_num_= bookH1("num","# rec electrons",20, 0.,20.,"N_{ele}");
//  h1_vertexP = bookH1("vertexP",        "ele p at vertex",       nbinp,0.,pmax,"p_{vertex} (GeV/c)");
//  h1_Et = bookH1("Et","ele SC transverse energy",  nbinpt,0.,ptmax,"E_{T} (GeV)");
//  h1_vertexTIP = bookH1("vertexTIP","ele transverse impact parameter (wrt bs)",90,0.,0.15,"TIP (cm)");
//  h1_charge = bookH1("charge","ele charge",5,-2.,2.,"charge");
  h1_vertexPt_barrel = bookH1("vertexPt_barrel","ele transverse momentum in barrel",nbinpt,0.,ptmax,"p_{T vertex} (GeV/c)");
  h1_vertexPt_endcaps = bookH1("vertexPt_endcaps","ele transverse momentum in endcaps",nbinpt,0.,ptmax,"p_{T vertex} (GeV/c)");
  h1_vertexEta = bookH1("vertexEta","ele momentum #eta",nbineta,etamin,etamax,"#eta");
//  h1_vertexPhi = bookH1("vertexPhi","ele  momentum #phi",nbinphi,phimin,phimax,"#phi (rad)");
  h2_vertexEtaVsPhi = bookH2("vertexEtaVsPhi","ele momentum #eta vs #phi",nbineta2D,etamin,etamax,nbinphi2D,phimin,phimax,"#eta","#phi (rad)");
//  h1_vertexX = bookH1("vertexX","ele vertex x",nbinxyz,-0.1,0.1,"x (cm)");
//  h1_vertexY = bookH1("vertexY","ele vertex y",nbinxyz,-0.1,0.1,"y (cm)");
  h2_vertexXvsY = bookH2("vertexXvsY","ele vertex x vs y",nbinxyz2D,-0.1,0.1,nbinxyz2D,-0.1,0.1,"x (cm)","y (cm)");
  h1_vertexZ = bookH1("vertexZ","ele vertex z",nbinxyz,-25, 25,"z (cm)");

  // super-clusters
//  h1_sclEn = bookH1("sclEnergy","ele supercluster energy",nbinp,0.,pmax,"E (GeV)");
//  h1_sclEta = bookH1("sclEta","ele supercluster #eta",nbineta,etamin,etamax,"#eta");
//  h1_sclPhi = bookH1("sclPhi","ele supercluster #phi",nbinphi,phimin,phimax,"#phi (rad)");
  h1_sclEt = bookH1("sclEt","ele supercluster transverse energy",nbinpt,0.,ptmax,"E_{T} (GeV)");

  // electron track
//  h1_ambiguousTracks = bookH1("ambiguousTracks","ele # ambiguous tracks",  5,0.,5.,"N_{amb. tk}");
//  h2_ambiguousTracksVsEta = bookH2("ambiguousTracksVsEta","ele # ambiguous tracks  vs #eta",  nbineta2D,etamin,etamax,5,0.,5.,"#eta","N_{amb. tk}");
//  h2_ambiguousTracksVsPhi = bookH2("ambiguousTracksVsPhi","ele # ambiguous tracks  vs #phi",  nbinphi2D,phimin,phimax,5,0.,5.,"#phi(rad)","N_{amb. tk}");
//  h2_ambiguousTracksVsPt = bookH2("ambiguousTracksVsPt","ele # ambiguous tracks vs pt",  nbinpt2D,0.,ptmax,5,0.,5.,"p_{T} (GeV/c),"N_{amb. tk}");
  h1_chi2 = bookH1("chi2","ele track #chi^{2}",100,0.,15.,"#Chi^{2}");
  py_chi2VsEta = bookP1("chi2VsEta","ele track #chi^{2} vs #eta",nbineta2D,etamin,etamax,0.,15.,"#eta","<#chi^{2}>");
  py_chi2VsPhi = bookP1("chi2VsPhi","ele track #chi^{2} vs #phi",nbinphi2D,phimin,phimax,0.,15.,"#phi (rad)","<#chi^{2}>");
  //h2_chi2VsPt = bookH2("chi2VsPt","ele track #chi^{2} vs pt",nbinpt2D,0.,ptmax,50,0.,15.,"p_{T} (GeV/c)","<#chi^{2}>");
  h1_foundHits = bookH1("foundHits","ele track # found hits",nbinfhits,0.,fhitsmax,"N_{hits}");
  py_foundHitsVsEta = bookP1("foundHitsVsEta","ele track # found hits vs #eta",nbineta2D,etamin,etamax,0.,fhitsmax,"#eta","<# hits>");
  py_foundHitsVsPhi = bookP1("foundHitsVsPhi","ele track # found hits vs #phi",nbinphi2D,phimin,phimax,0.,fhitsmax,"#phi (rad)","<# hits>");
//  h2_foundHitsVsPt = bookH2("foundHitsVsPt","ele track # found hits vs pt",nbinpt2D,0.,ptmax,nbinfhits,0.,fhitsmax,"p_{T} (GeV/c)","<# hits>");
  h1_lostHits = bookH1("lostHits","ele track # lost hits",5,0.,5.,"N_{lost hits}");
  py_lostHitsVsEta = bookP1("lostHitsVsEta","ele track # lost hits vs #eta",nbineta2D,etamin,etamax,0.,lhitsmax,"#eta","<# hits>");
  py_lostHitsVsPhi = bookP1("lostHitsVsPhi","ele track # lost hits vs #eta",nbinphi2D,phimin,phimax,0.,lhitsmax,"#phi (rad)","<# hits>");
//  h2_lostHitsVsPt = bookH2("lostHitsVsPt","ele track # lost hits vs #eta",nbinpt2D,0.,ptmax,nbinlhits,0.,lhitsmax,"p_{T} (GeV/c)","<# hits>");

  // electron matching and ID
  //h1_EoPout = bookH1( "EoPout","ele E/P_{out}",nbineop,0.,eopmax,"E_{seed}/P_{out}");
  //h1_dEtaCl_propOut = bookH1( "dEtaCl_propOut","ele #eta_{cl} - #eta_{tr}, prop from outermost",nbindetamatch,detamatchmin,detamatchmax,"#eta_{seedcl} - #eta_{tr}");
  //h1_dPhiCl_propOut = bookH1( "dPhiCl_propOut","ele #phi_{cl} - #phi_{tr}, prop from outermost",nbindphimatch,dphimatchmin,dphimatchmax,"#phi_{seedcl} - #phi_{tr} (rad)");
  //h1_outerP = bookH1( "outerP","ele track outer p, mean",nbinp,0.,pmax,"P_{out} (GeV/c)");
  //h1_outerP_mode = bookH1( "outerP_mode","ele track outer p, mode",nbinp,0.,pmax,"P_{out} (GeV/c)");
  h1_Eop_barrel = bookH1( "Eop_barrel","ele E/P_{vertex} in barrel",nbineop,0.,eopmax,"E/P_{vertex}");
  h1_Eop_endcaps = bookH1( "Eop_endcaps","ele E/P_{vertex} in endcaps",nbineop,0.,eopmax,"E/P_{vertex}");
  py_EopVsPhi = bookP1("EopVsPhi","ele E/P_{vertex} vs #phi",nbinphi2D,phimin,phimax,0.,eopmax,"#phi (rad)","<E/P_{vertex}>");
//  h2_EopVsPt = bookH2("EopVsPt","ele E/P_{vertex} vs pt",nbinpt2D,0.,ptmax,nbineop,0.,eopmax,"p_{T} (GeV/c)","E/P_{vertex}");
  h1_EeleOPout_barrel = bookH1( "EeleOPout_barrel","ele E_{ele}/P_{out} in barrel",nbineop,0.,eopmax,"E_{ele}/P_{out}");
  h1_EeleOPout_endcaps = bookH1( "EeleOPout_endcaps","ele E_{ele}/P_{out} in endcaps",nbineop,0.,eopmax,"E_{ele}/P_{out}");
//  h2_EeleOPoutVsPhi = bookH2("EeleOPoutVsPhi","ele E_{ele}/P_{out} vs #phi",nbinphi2D,phimin,phimax,nbineop,0.,eopmax,"#phi","E_{ele}/P_{out}");
//  h2_EeleOPoutVsPt = bookH2("EeleOPoutVsPt","ele E_{ele}/P_{out} vs pt",nbinpt2D,0.,ptmax,nbineop,0.,eopmax,"p_{T} (GeV/c)","E_{ele}/P_{out}");
  h1_dEtaSc_propVtx_barrel = bookH1( "dEtaSc_propVtx_barrel","ele #eta_{sc} - #eta_{tr}, prop from vertex, in barrel",nbindetamatch,detamatchmin,detamatchmax,"#eta_{sc} - #eta_{tr}");
  h1_dEtaSc_propVtx_endcaps = bookH1( "dEtaSc_propVtx_endcaps","ele #eta_{sc} - #eta_{tr}, prop from vertex, in endcaps",nbindetamatch,detamatchmin,detamatchmax,"#eta_{sc} - #eta_{tr}");
  py_dEtaSc_propVtxVsPhi = bookP1("dEtaSc_propVtxVsPhi","ele #eta_{sc} - #eta_{tr}, prop from vertex vs #phi",nbinphi2D,phimin,phimax,detamatchmin,detamatchmax,"#phi (rad)","<#eta_{sc} - #eta_{tr}>");
//  h2_dEtaSc_propVtxVsPt = bookH2("dEtaSc_propVtxVsPt","ele #eta_{sc} - #eta_{tr}, prop from vertex vs pt",nbinpt2D,0.,ptmax,nbindetamatch,detamatchmin,detamatchmax,"#eta_{sc} - #eta_{tr}");
  h1_dEtaEleCl_propOut_barrel = bookH1( "dEtaEleCl_propOut_barrel","ele #eta_{EleCl} - #eta_{tr}, prop from outermost, in barrel",nbindetamatch,detamatchmin,detamatchmax,"#eta_{elecl} - #eta_{tr}");
  h1_dEtaEleCl_propOut_endcaps = bookH1( "dEtaEleCl_propOut_endcaps","ele #eta_{EleCl} - #eta_{tr}, prop from outermost, in endcaps",nbindetamatch,detamatchmin,detamatchmax,"#eta_{elecl} - #eta_{tr}");
//  h2_dEtaEleCl_propOutVsPhi = bookH2("dEtaEleCl_propOutVsPhi","ele #eta_{EleCl} - #eta_{tr}, prop from outermost vs #phi",nbinphi2D,phimin,phimax,nbindetamatch,detamatchmin,detamatchmax,"#phi (rad)","#eta_{elecl} - #eta_{tr}");
//  h2_dEtaEleCl_propOutVsPt = bookH2("dEtaEleCl_propOutVsPt","ele #eta_{EleCl} - #eta_{tr}, prop from outermost vs pt",nbinpt2D,0.,ptmax,nbindetamatch,detamatchmin,detamatchmax,"p_{T} (GeV/c)","#eta_{elecl} - #eta_{tr}");
  h1_dPhiSc_propVtx_barrel = bookH1( "dPhiSc_propVtx_barrel","ele #phi_{sc} - #phi_{tr}, prop from vertex, in barrel",nbindphimatch,dphimatchmin,dphimatchmax,"#phi_{sc} - #phi_{tr} (rad)");
  h1_dPhiSc_propVtx_endcaps = bookH1( "dPhiSc_propVtx_endcaps","ele #phi_{sc} - #phi_{tr}, prop from vertex, in endcaps",nbindphimatch,dphimatchmin,dphimatchmax,"#phi_{sc} - #phi_{tr} (rad)");
  py_dPhiSc_propVtxVsPhi = bookP1("dPhiSc_propVtxVsPhi","ele #phi_{sc} - #phi_{tr}, prop from vertex vs #phi",nbinphi2D,phimin,phimax,dphimatchmin,dphimatchmax,"#phi (rad)","<#phi_{sc} - #phi_{tr}> (rad)");
//  h2_dPhiSc_propVtxVsPt = bookH2("dPhiSc_propVtxVsPt","ele #phi_{sc} - #phi_{tr}, prop from vertex vs pt",nbinpt2D,0.,ptmax,nbindphimatch,dphimatchmin,dphimatchmax,"p_{T} (GeV/c)","#phi_{sc} - #phi_{tr} (rad)");
  h1_dPhiEleCl_propOut_barrel = bookH1( "dPhiEleCl_propOut_barrel","ele #phi_{EleCl} - #phi_{tr}, prop from outermost, in barrel",nbindphimatch,dphimatchmin,dphimatchmax,"#phi_{elecl} - #phi_{tr} (rad)");
  h1_dPhiEleCl_propOut_endcaps = bookH1( "dPhiEleCl_propOut_endcaps","ele #phi_{EleCl} - #phi_{tr}, prop from outermost, in endcaps",nbindphimatch,dphimatchmin,dphimatchmax,"#phi_{elecl} - #phi_{tr} (rad)");
//  h2_dPhiEleCl_propOutVsPhi = bookH2("dPhiEleCl_propOutVsPhi","ele #phi_{EleCl} - #phi_{tr}, prop from outermost vs #phi",nbinphi2D,phimin,phimax,nbindphimatch,dphimatchmin,dphimatchmax,"#phi_{elecl} - #phi_{tr} (rad)");
//  h2_dPhiEleCl_propOutVsPt = bookH2("dPhiEleCl_propOutVsPt","ele #phi_{EleCl} - #phi_{tr}, prop from outermost vs pt",nbinpt2D,0.,ptmax,nbindphimatch,dphimatchmin,dphimatchmax,"p_{T} (GeV/c)","#phi_{elecl} - #phi_{tr} (rad)");
  h1_Hoe_barrel = bookH1("Hoe_barrel","ele hadronic energy / em energy, in barrel", nbinhoe, hoemin, hoemax,"H/E","Events","ELE_LOGY E1 P") ;
  h1_Hoe_endcaps = bookH1("Hoe_endcaps","ele hadronic energy / em energy, in endcaps", nbinhoe, hoemin, hoemax,"H/E","Events","ELE_LOGY E1 P") ;
  py_HoeVsPhi = bookP1("HoeVsPhi","ele hadronic energy / em energy vs #phi",nbinphi2D,phimin,phimax,hoemin,hoemax,"#phi (rad)","<H/E>","E1 P") ;
//  h2_HoeVsPt = bookH2("HoeVsPt","ele hadronic energy / em energy vs pt",nbinpt2D,0.,ptmax,nbinhoe,hoemin,hoemax,"p_{T} (GeV/c)","<H/E>","ELE_LOGY COLZ") ;
  h1_sclSigEtaEta_barrel = bookH1("sclSigEtaEta_barrel","ele supercluster sigma ieta ieta in barrel",100,0.,0.05,"sietaieta");
  h1_sclSigEtaEta_endcaps = bookH1("sclSigEtaEta_endcaps","ele supercluster sigma ieta ieta in endcaps",100,0.,0.05,"sietaieta");

  // fbrem
//  h1_innerPt_mean = bookH1( "innerPt_mean","ele track inner p_{T}, mean",nbinpt,0.,ptmax,"P_{T in} (GeV/c)");
//  h1_outerPt_mean = bookH1( "outerPt_mean","ele track outer p_{T}, mean",nbinpt,0.,ptmax,"P_{T out} (GeV/c)");
//  h1_outerPt_mode = bookH1( "outerPt_mode","ele track outer p_{T}, mode",nbinpt,0.,ptmax,"P_{T out} (GeV/c)");
//  //h_PinMnPout_mode = bookH1( "PinMnPout_mode","ele track inner p - outer p, mode"   ,nbinp,0.,100.,"P_{in} - P_{out} (GeV/c)");
//  h1_PinMnPout = bookH1( "PinMnPout","ele track inner p - outer p, mean" ,nbinp,0.,200.,"P_{in} - P_{out} (GeV/c)");
//  h1_PinMnPout_mode = bookH1( "PinMnPout_mode","ele track inner p - outer p, mode",nbinp,0.,100.,"P_{in} - P_{out}, mode (GeV/c)");
  h1_fbrem = bookH1("fbrem","ele brem fraction",100,0.,1.,"P_{in} - P_{out} / P_{in}") ;
  py_fbremVsEta = bookP1("fbremVsEta","ele brem fraction vs #eta",nbineta2D,etamin,etamax,0.,1.,"#eta","<P_{in} - P_{out} / P_{in}>") ;
  py_fbremVsPhi = bookP1("fbremVsPhi","ele brem fraction vs #phi",nbinphi2D,phimin,phimax,0.,1.,"#phi (rad)","<P_{in} - P_{out} / P_{in}>") ;
//  h2_fbremVsPt = bookH2("fbremVsPt","ele brem fraction vs pt",nbinpt2D,0.,ptmax,100,0.,1.,"p_{T} (GeV/c)","<P_{in} - P_{out} / P_{in}>") ;
  h1_classes = bookH1("classes","ele electron classes",10,0.0,10.);

  // pflow
  h1_mva = bookH1( "mva","ele identification mva",100,-1.,1.,"mva");
  h1_provenance = bookH1( "provenance","ele provenance",5,-2.,3.,"provenance");

  // isolation
  h1_tkSumPt_dr03 = bookH1("tkSumPt_dr03","tk isolation sum, dR=0.3",100,0.0,20.,"TkIsoSum (GeV/c)","Events","ELE_LOGY E1 P");
  h1_ecalRecHitSumEt_dr03 = bookH1("ecalRecHitSumEt_dr03","ecal isolation sum, dR=0.3",100,0.0,20.,"EcalIsoSum (GeV)","Events","ELE_LOGY E1 P");
  h1_hcalTowerSumEt_dr03 = bookH1("hcalTowerSumEt_dr03","hcal isolation sum, dR=0.3",100,0.0,20.,"HcalIsoSum (GeV)","Events","ELE_LOGY E1 P");
//  h1_hcalDepth1TowerSumEt_dr03 = bookH1("hcalDepth1TowerSumEt_dr03","hcal depth1 isolation sum, dR=0.3",100,0.0,20.,"Hcal1IsoSum (GeV)","Events","ELE_LOGY E1 P");
//  h1_hcalDepth2TowerSumEt_dr03 = bookH1("hcalDepth2TowerSumEt_dr03","hcal depth2 isolation sum, dR=0.3",100,0.0,20.,"Hcal2IsoSum (GeV)","Events","ELE_LOGY E1 P");
//  h1_tkSumPt_dr04 = bookH1("tkSumPt_dr04","hcal isolation sum",100,0.0,20.,"TkIsoSum (GeV/c)","Events","ELE_LOGY E1 P");
//  h1_ecalRecHitSumEt_dr04 = bookH1("ecalRecHitSumEt_dr04","ecal isolation sum, dR=0.4",100,0.0,20.,"EcalIsoSum (GeV)","Events","ELE_LOGY E1 P");
//  h1_hcalTowerSumEt_dr04 = bookH1("hcalTowerSumEt_dr04","hcal isolation sum, dR=0.4",100,0.0,20.,"HcalIsoSum (GeV)","Events","ELE_LOGY E1 P");
////  h1_hcalDepth1TowerSumEt_dr04 = bookH1("hcalDepth1TowerSumEt_dr04","hcal depth1 isolation sum, dR=0.4",100,0.0,20.,"Hcal1IsoSum (GeV)","Events","ELE_LOGY E1 P");
////  h1_hcalDepth2TowerSumEt_dr04 = bookH1("hcalDepth2TowerSumEt_dr04","hcal depth2 isolation sum, dR=0.4",100,0.0,20.,"Hcal2IsoSum (GeV)","Events","ELE_LOGY E1 P");

  // di-electron mass
  setBookIndex(200) ;
  h1_mee = bookH1("mee","ele pairs invariant mass", nbinmee, meemin, meemax,"m_{ee} (GeV/c^{2})");
  h1_mee_os = bookH1("mee_os","ele pairs invariant mass, opposite sign", nbinmee, meemin, meemax,"m_{e^{+}e^{-}} (GeV/c^{2})");



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
//  std::string htitle = "# "+matchingObjectType+"s", xtitle = "N_{"+matchingObjectType+"}" ;
//  h1_matchingObject_Num = bookH1withSumw2("matchingObject_Num",htitle,nbinfhits,0.,fhitsmax,xtitle) ;

  // matching object distributions
  h1_matchingObject_Eta = bookH1withSumw2("matchingObject_Eta",matchingObjectType+" #eta",nbineta,etamin,etamax,"#eta_{SC}");
//  h1_matchingObject_AbsEta = bookH1withSumw2("matchingObject_AbsEta",matchingObjectType+" |#eta|",nbineta/2,0.,etamax,"|#eta|_{SC}");
//  h1_matchingObject_P = bookH1withSumw2("matchingObject_P",matchingObjectType+" p",nbinp,0.,pmax,"E_{SC} (GeV)");
  h1_matchingObject_Pt = bookH1withSumw2("matchingObject_Pt",matchingObjectType+" pt",nbinpteff,5.,ptmax,"pt_{SC} (GeV/c)");
  h1_matchingObject_Phi = bookH1withSumw2("matchingObject_Phi",matchingObjectType+" #phi",nbinphi,phimin,phimax,"#phi (rad)");
//  h1_matchingObject_Z = bookH1withSumw2("matchingObject_Z",matchingObjectType+" z",nbinxyz,-25,25,"z (cm)");

  h1_matchedObject_Eta = bookH1withSumw2("matchedObject_Eta","Efficiency vs matching SC #eta",nbineta,etamin,etamax,"#eta_{SC}");
//  h1_matchedObject_AbsEta = bookH1withSumw2("matchedObject_AbsEta","Efficiency vs matching SC |#eta|",nbineta/2,0.,2.5,"|#eta|_{SC}");
  h1_matchedObject_Pt = bookH1withSumw2("matchedObject_Pt","Efficiency vs matching SC E_{T}",nbinpteff,5.,ptmax,"pt_{SC} (GeV/c)");
  h1_matchedObject_Phi = bookH1withSumw2("matchedObject_Phi","Efficiency vs matching SC #phi",nbinphi,phimin,phimax,"#phi (rad)");
//  h1_matchedObject_Z = bookH1withSumw2("matchedObject_Z","Efficiency vs matching SC z",nbinxyz,-25,25,"z (cm)");

//  // classes
//  h1_matchedEle_eta = bookH1( "matchedEle_eta", "ele electron #eta",  nbineta/2,0.0,etamax,"#eta");
//  h1_matchedEle_eta_golden = bookH1( "matchedEle_eta_golden", "ele electron #eta golden",  nbineta/2,0.0,etamax,"#eta");
//  h1_matchedEle_eta_shower = bookH1( "matchedEle_eta_shower", "ele electron #eta showering",  nbineta/2,0.0,etamax,"#eta");
//  //h1_matchedEle_eta_bbrem = bookH1( "matchedEle_eta_bbrem", "ele electron #eta bbrem",  nbineta/2,0.0,etamax,"#eta");
//  //h1_matchedEle_eta_narrow = bookH1( "matchedEle_eta_narrow", "ele electron #eta narrow",  nbineta/2,0.0,etamax,"#eta");
//
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

  int ievt = iEvent.id().event();
  int irun = iEvent.id().run();
  int ils = iEvent.luminosityBlock();

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

    // basic quantities
    if (gsfIter->isEB()) h1_vertexPt_barrel->Fill( gsfIter->pt() );
    if (gsfIter->isEE()) h1_vertexPt_endcaps->Fill( gsfIter->pt() );
    h1_vertexEta->Fill( gsfIter->eta() );
//    h1_vertexPhi->Fill( gsfIter->phi() );
    h2_vertexEtaVsPhi->Fill( gsfIter->eta(), gsfIter->phi() );
//    h1_vertexX->Fill( gsfIter->vertex().x() );
//    h1_vertexY->Fill( gsfIter->vertex().y() );
    h2_vertexXvsY->Fill( gsfIter->vertex().x(), gsfIter->vertex().y() );
    h1_vertexZ->Fill( gsfIter->vertex().z() );
//    h1_vertexP->Fill( gsfIter->p() );
//    h1_Et->Fill( gsfIter->superCluster()->energy()/cosh( gsfIter->superCluster()->eta()) );
//    h1_vertexTIP->Fill( vertexTIP );
//    h1_charge->Fill( gsfIter->charge() );

    // supercluster related distributions
    reco::SuperClusterRef sclRef = gsfIter->superCluster() ;
    // ALREADY DONE IN GSF ELECTRON CORE
    //    if (!gsfIter->ecalDrivenSeed()&&gsfIter->trackerDrivenSeed())
    //      sclRef = gsfIter->parentSuperCluster() ;
//    h1_sclEn->Fill(sclRef->energy());
//    h1_sclEta->Fill(sclRef->eta());
//    h1_sclPhi->Fill(sclRef->phi());
    double R=TMath::Sqrt(sclRef->x()*sclRef->x() + sclRef->y()*sclRef->y() +sclRef->z()*sclRef->z());
    double Rt=TMath::Sqrt(sclRef->x()*sclRef->x() + sclRef->y()*sclRef->y());
    h1_sclEt->Fill(sclRef->energy()*(Rt/R));

    // track related distributions
//    h1_ambiguousTracks->Fill( gsfIter->ambiguousGsfTracksSize() );
//    h2_ambiguousTracksVsEta->Fill( gsfIter->eta(), gsfIter->ambiguousGsfTracksSize() );
//    h2_ambiguousTracksVsPhi->Fill( gsfIter->phi(), gsfIter->ambiguousGsfTracksSize() );
//    h2_ambiguousTracksVsPt->Fill( gsfIter->pt(), gsfIter->ambiguousGsfTracksSize() );
    if (!readAOD_)
     { // track extra does not exist in AOD
      h1_foundHits->Fill( gsfIter->gsfTrack()->numberOfValidHits() );
      py_foundHitsVsEta->Fill( gsfIter->eta(), gsfIter->gsfTrack()->numberOfValidHits() );
      py_foundHitsVsPhi->Fill( gsfIter->phi(), gsfIter->gsfTrack()->numberOfValidHits() );
      //h2_foundHitsVsPt->Fill( gsfIter->pt(), gsfIter->gsfTrack()->numberOfValidHits() );
      h1_lostHits->Fill( gsfIter->gsfTrack()->numberOfLostHits() );
      py_lostHitsVsEta->Fill( gsfIter->eta(), gsfIter->gsfTrack()->numberOfLostHits() );
      py_lostHitsVsPhi->Fill( gsfIter->phi(), gsfIter->gsfTrack()->numberOfLostHits() );
      //h2_lostHitsVsPt->Fill( gsfIter->pt(), gsfIter->gsfTrack()->numberOfLostHits() );
      h1_chi2->Fill( gsfIter->gsfTrack()->normalizedChi2() );
      py_chi2VsEta->Fill( gsfIter->eta(), gsfIter->gsfTrack()->normalizedChi2() );
      py_chi2VsPhi->Fill( gsfIter->phi(), gsfIter->gsfTrack()->normalizedChi2() );
      //h2_chi2VsPt->Fill( gsfIter->pt(), gsfIter->gsfTrack()->normalizedChi2() );
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
     }
    if (gsfIter->isEE())
     {
      h1_Eop_endcaps->Fill( gsfIter->eSuperClusterOverP() );
      h1_EeleOPout_endcaps->Fill( gsfIter->eEleClusterOverPout() );
      h1_dEtaSc_propVtx_endcaps->Fill(gsfIter->deltaEtaSuperClusterTrackAtVtx());
      h1_dEtaEleCl_propOut_endcaps->Fill(gsfIter->deltaEtaEleClusterTrackAtCalo());
      h1_dPhiSc_propVtx_endcaps->Fill(gsfIter->deltaPhiSuperClusterTrackAtVtx());
      h1_dPhiEleCl_propOut_endcaps->Fill(gsfIter->deltaPhiEleClusterTrackAtCalo());
      h1_Hoe_endcaps->Fill(gsfIter->hadronicOverEm());
      h1_sclSigEtaEta_endcaps->Fill( gsfIter->scSigmaEtaEta() );
     }
    py_EopVsPhi->Fill( gsfIter->phi(), gsfIter->eSuperClusterOverP() );
//    h2_EopVsPt->Fill( gsfIter->pt(), gsfIter->eSuperClusterOverP() );
//    h2_EeleOPoutVsPhi->Fill( gsfIter->phi(), gsfIter->eEleClusterOverPout() );
//    h2_EeleOPoutVsPt->Fill( gsfIter->pt(), gsfIter->eEleClusterOverPout() );
    py_dEtaSc_propVtxVsPhi->Fill(gsfIter->phi(), gsfIter->deltaEtaSuperClusterTrackAtVtx());
//    h2_dEtaSc_propVtxVsPt->Fill(gsfIter->pt(), gsfIter->deltaEtaSuperClusterTrackAtVtx());
//    h2_dEtaEleCl_propOutVsPhi->Fill(gsfIter->phi(), gsfIter->deltaEtaEleClusterTrackAtCalo());
//    h2_dEtaEleCl_propOutVsPt->Fill(gsfIter->pt(), gsfIter->deltaEtaEleClusterTrackAtCalo());
    py_dPhiSc_propVtxVsPhi->Fill(gsfIter->phi(), gsfIter->deltaPhiSuperClusterTrackAtVtx());
//    h2_dPhiSc_propVtxVsPt->Fill(gsfIter->pt(), gsfIter->deltaPhiSuperClusterTrackAtVtx());
//    h2_dPhiEleCl_propOutVsPhi->Fill(gsfIter->phi(), gsfIter->deltaPhiEleClusterTrackAtCalo());
//    h2_dPhiEleCl_propOutVsPt->Fill(gsfIter->pt(), gsfIter->deltaPhiEleClusterTrackAtCalo());
    py_HoeVsPhi->Fill(gsfIter->phi(), gsfIter->hadronicOverEm());
//    h2_HoeVsPt->Fill(gsfIter->pt(), gsfIter->hadronicOverEm());

//    // from gsf track interface, hence using mean
//    if (!readAOD_)
//     { // track extra does not exist in AOD
//      h_PinMnPout->Fill( gsfIter->gsfTrack()->innerMomentum().R() - gsfIter->gsfTrack()->outerMomentum().R() );
//      //h_outerP->Fill( gsfIter->gsfTrack()->outerMomentum().R() );
//      h_innerPt_mean->Fill( gsfIter->gsfTrack()->innerMomentum().Rho() );
//      h_outerPt_mean->Fill( gsfIter->gsfTrack()->outerMomentum().Rho() );
//     }
//
//    // from electron interface, hence using mode
//    h_PinMnPout_mode->Fill( gsfIter->trackMomentumAtVtx().R() - gsfIter->trackMomentumOut().R() );
//    //h_outerP_mode->Fill( gsfIter->trackMomentumOut().R() );
//    h_outerPt_mode->Fill( gsfIter->trackMomentumOut().Rho() );
//

    // fbrem, classes
    h1_fbrem->Fill(gsfIter->fbrem()) ;
    py_fbremVsEta->Fill(gsfIter->eta(),gsfIter->fbrem()) ;
    py_fbremVsPhi->Fill(gsfIter->phi(),gsfIter->fbrem()) ;
//    h2_fbremVsPt->Fill(gsfIter->pt(),gsfIter->fbrem()) ;
    int eleClass = gsfIter->classification() ;
    if (gsfIter->isEE()) eleClass+=5;
    h1_classes->Fill(eleClass) ;


    // pflow
    h1_mva->Fill(gsfIter->mva()) ;
    if (gsfIter->ecalDrivenSeed()) h1_provenance->Fill(1.) ;
    if (gsfIter->trackerDrivenSeed()) h1_provenance->Fill(-1.) ;
    if (gsfIter->trackerDrivenSeed()||gsfIter->ecalDrivenSeed()) h1_provenance->Fill(0.);
    if (gsfIter->trackerDrivenSeed()&&!gsfIter->ecalDrivenSeed()) h1_provenance->Fill(-2.);
    if (!gsfIter->trackerDrivenSeed()&&gsfIter->ecalDrivenSeed()) h1_provenance->Fill(2.);

    // isolation
    h1_tkSumPt_dr03->Fill(gsfIter->dr03TkSumPt());
    h1_ecalRecHitSumEt_dr03->Fill(gsfIter->dr03EcalRecHitSumEt());
    h1_hcalTowerSumEt_dr03->Fill(gsfIter->dr03HcalTowerSumEt());
//    h1_hcalDepth1TowerSumEt_dr03->Fill(gsfIter->dr03HcalDepth1TowerSumEt());
//    h1_hcalDepth2TowerSumEt_dr03->Fill(gsfIter->dr03HcalDepth2TowerSumEt());
//    h1_tkSumPt_dr04->Fill(gsfIter->dr04TkSumPt());
//    h1_ecalRecHitSumEt_dr04->Fill(gsfIter->dr04EcalRecHitSumEt());
//    h1_hcalTowerSumEt_dr04->Fill(gsfIter->dr04HcalTowerSumEt());
////    h1_hcalDepth1TowerSumEt_dr04->Fill(gsfIter->dr04HcalDepth1TowerSumEt());
////    h1_hcalDepth2TowerSumEt_dr04->Fill(gsfIter->dr04HcalDepth2TowerSumEt());

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
//    //if (std::abs(moIter->eta()) > 1.5) continue;
//    // select central z
//    //if ( std::abs((*mcIter)->production_vertex()->position().z())>50.) continue;

    h1_matchingObject_Eta->Fill( moIter->eta() );
//    h1_matchingObject_AbsEta->Fill( std::abs(moIter->eta()) );
//    h1_matchingObject_P->Fill( moIter->energy() );
    h1_matchingObject_Pt->Fill( moIter->energy()/cosh(moIter->eta()) );
    h1_matchingObject_Phi->Fill( moIter->phi() );
//    h1_matchingObject_Z->Fill(  moIter->z() );

    bool okGsfFound = false ;
    double gsfOkRatio = 999999999. ;
    reco::GsfElectron bestGsfElectron ;
    reco::GsfElectronCollection::const_iterator gsfIter ;
    for
     ( gsfIter=gsfElectrons->begin() ;
       gsfIter!=gsfElectrons->end() ;
       gsfIter++ )
     {
      reco::GsfElectronCollection::const_iterator gsfIter2 ;
      for
       ( gsfIter2=gsfIter+1;
         gsfIter2!=gsfElectrons->end() ;
         gsfIter2++ )
       {
        float invMass = computeInvMass(*gsfIter,*gsfIter2) ;
        if(matchingObjectNum == 1){h1_mee->Fill(invMass) ;}
        if ((matchingObjectNum == 1) && (((gsfIter->charge())*(gsfIter2->charge()))<0.))
         { h1_mee_os->Fill(invMass) ; }
       }

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
          //if ( (genPc->pdg_id() == 11) && (gsfIter->charge() < 0.) || (genPc->pdg_id() == -11) &&
          //(gsfIter->charge() > 0.) ){
          double tmpGsfRatio = gsfIter->p()/moIter->energy() ;
          if ( std::abs(tmpGsfRatio-1) < std::abs(gsfOkRatio-1) )
           {
            gsfOkRatio = tmpGsfRatio;
            bestGsfElectron=*gsfIter;
            okGsfFound = true;
           }
          //}
         }
       }
     } // loop over rec ele to look for the best one
    if (okGsfFound)
     {
      // generated distributions for matched electrons
      h1_matchedObject_Eta->Fill( moIter->eta() );
    //  h1_matchedObject_AbsEta->Fill( std::abs(moIter->eta()) );
      h1_matchedObject_Pt->Fill( moIter->energy()/cosh(moIter->eta()) );
      h1_matchedObject_Phi->Fill( moIter->phi() );
    //  h1_matchedObject_Z->Fill( moIter->z() );

      //classes
    //  int eleClass = bestGsfElectron.classification() ;
    //  h_classes->Fill(eleClass) ;
    //  h_matchedEle_eta->Fill(std::abs(bestGsfElectron.eta()));
    //  if (bestGsfElectron.classification() == GsfElectron::GOLDEN) h_matchedEle_eta_golden->Fill(std::abs(bestGsfElectron.eta()));
    //  if (bestGsfElectron.classification() == GsfElectron::SHOWERING) h_matchedEle_eta_shower->Fill(std::abs(bestGsfElectron.eta()));
    //  //if (bestGsfElectron.classification() == GsfElectron::BIGBREM) h_matchedEle_eta_bbrem->Fill(std::abs(bestGsfElectron.eta()));
    //  //if (bestGsfElectron.classification() == GsfElectron::OLDNARROW) h_matchedEle_eta_narrow->Fill(std::abs(bestGsfElectron.eta()));
     }

   } // loop overmatching object

//  h_matchingObject_Num->Fill(matchingObjectNum) ;

 }

float ElectronAnalyzer::computeInvMass
 ( const reco::GsfElectron & e1,
   const reco::GsfElectron & e2 )
 {
  math::XYZTLorentzVector p12 = e1.p4()+e2.p4() ;
  float mee2 = p12.Dot(p12) ;
  float invMass = sqrt(mee2) ;
  return invMass ;
 }

//bool ElectronAnalyzer::trigger( const edm::Event & e )
// {
//  // retreive TriggerResults from the event
//  edm::Handle<edm::TriggerResults> triggerResults ;
//  e.getByLabel(triggerResults_,triggerResults) ;
//
//  bool accept = false ;
//
//  if (triggerResults.isValid())
//   {
//    //std::cout << "TriggerResults found, number of HLT paths: " << triggerResults->size() << std::endl;
//    // get trigger names
//    const edm::TriggerNames & triggerNames_ = e.triggerNames(*triggerResults);
////    if (nEvents_==1)
////     {
////      for (unsigned int i=0; i<triggerNames_.size(); i++)
////       { std::cout << "trigger path= " << triggerNames_.triggerName(i) << std::endl; }
////     }
//
//    unsigned int n = HLTPathsByName_.size() ;
//    for (unsigned int i=0; i!=n; i++)
//     {
//      HLTPathsByIndex_[i]=triggerNames_.triggerIndex(HLTPathsByName_[i]) ;
//     }
//
//    // empty input vectors (n==0) means any trigger paths
//    if (n==0)
//     {
//      n=triggerResults->size() ;
//      HLTPathsByName_.resize(n) ;
//      HLTPathsByIndex_.resize(n) ;
//      for ( unsigned int i=0 ; i!=n ; i++)
//       {
//        HLTPathsByName_[i]=triggerNames_.triggerName(i) ;
//        HLTPathsByIndex_[i]=i ;
//       }
//     }
//
////    if (nEvents_==1)
////     {
////      if (n>0)
////       {
////        std::cout << "HLT trigger paths requested: index, name and valididty:" << std::endl;
////        for (unsigned int i=0; i!=n; i++)
////         {
////          bool validity = HLTPathsByIndex_[i]<triggerResults->size();
////          std::cout
////            << " " << HLTPathsByIndex_[i]
////            << " " << HLTPathsByName_[i]
////            << " " << validity << std::endl;
////         }
////       }
////     }
//
//    // count number of requested HLT paths which have fired
//    unsigned int fired=0 ;
//    for ( unsigned int i=0 ; i!=n ; i++ )
//     {
//      if (HLTPathsByIndex_[i]<triggerResults->size())
//       {
//        if (triggerResults->accept(HLTPathsByIndex_[i]))
//         {
//          fired++ ;
//          h1_triggers->Fill(float(HLTPathsByIndex_[i]));
//          //std::cout << "Fired HLT path= " << HLTPathsByName_[i] << std::endl ;
//          accept = true ;
//         }
//       }
//     }
//
//   }
//
//  return accept ;
// }

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
