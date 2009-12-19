
#include "DQMOffline/EGamma/interface/ElectronAnalyzer.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "FWCore/Framework/interface/TriggerNames.h"
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
  electronCollection_=conf.getParameter<edm::InputTag>("ElectronCollection");
  matchingObjectCollection_ = conf.getParameter<edm::InputTag>("MatchingObjectCollection");
  matchingCondition_ = conf.getParameter<std::string>("MatchingCondition");
  readAOD_ = conf.getParameter<bool>("ReadAOD");

  assert (matchingCondition_=="Cone") ;

  maxPtMatchingObject_ = conf.getParameter<double>("MaxPtMatchingObject");
  maxAbsEtaMatchingObject_ = conf.getParameter<double>("MaxAbsEtaMatchingObject");
  deltaR_ = conf.getParameter<double>("DeltaR");

  Selection_ = conf.getParameter<int>("Selection");
  massLow_ = conf.getParameter< double >("MassLow");
  massHigh_ = conf.getParameter< double >("MassHigh");
  TPchecksign_ = conf.getParameter<bool>("TpCheckSign");
  TAGcheckclass_ = conf.getParameter<bool>("TagCheckClass");
  PROBEetcut_ = conf.getParameter<bool>("ProbeEtCut");
  PROBEcheckclass_ = conf.getParameter<bool>("ProbeCheckClass");

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

  triggerResults_ = conf.getParameter<edm::InputTag>("TriggerResults");
  HLTPathsByName_= conf.getParameter<std::vector<std::string > >("HltPaths");
  HLTPathsByIndex_.resize(HLTPathsByName_.size());

  etamin=conf.getParameter<double>("EtaMin");
  etamax=conf.getParameter<double>("EtaMax");
  phimin=conf.getParameter<double>("PhiMin");
  phimax=conf.getParameter<double>("PhiMax");
  ptmax=conf.getParameter<double>("PtMax");
  pmax=conf.getParameter<double>("PMax");
  eopmax=conf.getParameter<double>("EopMax");
  eopmaxsht=conf.getParameter<double>("EopMaxSht");
  detamin=conf.getParameter<double>("DetaMin");
  detamax=conf.getParameter<double>("DetaMax");
  dphimin=conf.getParameter<double>("DphiMin");
  dphimax=conf.getParameter<double>("DphiMax");
  detamatchmin=conf.getParameter<double>("DetaMatchMin");
  detamatchmax=conf.getParameter<double>("DetaMatchMax");
  dphimatchmin=conf.getParameter<double>("DphiMatchMin");
  dphimatchmax=conf.getParameter<double>("DphiMatchMax");
  fhitsmax=conf.getParameter<double>("FhitsMax");
  lhitsmax=conf.getParameter<double>("LhitsMax");
  nbineta=conf.getParameter<int>("NbinEta");
  nbineta2D=conf.getParameter<int>("NbinEta2D");
  nbinp=conf.getParameter<int>("NbinP");
  nbinpt=conf.getParameter<int>("NbinPt");
  nbinp2D=conf.getParameter<int>("NbinP2D");
  nbinpt2D=conf.getParameter<int>("NbinPt2D");
  nbinpteff=conf.getParameter<int>("NbinPtEff");
  nbinphi=conf.getParameter<int>("NbinPhi");
  nbinphi2D=conf.getParameter<int>("NbinPhi2D");
  nbineop=conf.getParameter<int>("NbinEop");
  nbineop2D=conf.getParameter<int>("NbinEop2D");
  nbinfhits=conf.getParameter<int>("NbinFhits");
  nbinlhits=conf.getParameter<int>("NbinLhits");
  nbinxyz=conf.getParameter<int>("NbinXyz");
  nbindeta=conf.getParameter<int>("NbinDeta");
  nbindphi=conf.getParameter<int>("NbinDphi");
  nbindetamatch=conf.getParameter<int>("NbinDetaMatch");
  nbindphimatch=conf.getParameter<int>("NbinDphiMatch");
  nbindetamatch2D=conf.getParameter<int>("NbinDetaMatch2D");
  nbindphimatch2D=conf.getParameter<int>("NbinDphiMatch2D");
  nbinpoptrue= conf.getParameter<int>("NbinPopTrue");
  poptruemin=conf.getParameter<double>("PopTrueMin");
  poptruemax=conf.getParameter<double>("PopTrueMax");
  nbinmee= conf.getParameter<int>("NbinMee");
  meemin=conf.getParameter<double>("MeeMin");
  meemax=conf.getParameter<double>("MeeMax");
  nbinhoe= conf.getParameter<int>("NbinHoe");
  hoemin=conf.getParameter<double>("HoeMin");
  hoemax=conf.getParameter<double>("HoeMax");
 }

ElectronAnalyzer::~ElectronAnalyzer()
 {}

void ElectronAnalyzer::book()
 {
  nEvents_ = 0 ;
  nAfterTrigger_ = 0 ;

  // matched electrons ?
  h_ele_vertexPt = bookH1( "h_ele_vertexPt","ele transverse momentum",nbinpt,0.,ptmax,"p_{T vertex} (GeV/c)");
  h_ele_vertexEta = bookH1( "h_ele_vertexEta","ele momentum eta",nbineta,etamin,etamax,"#eta");
  h_ele_vertexPhi = bookH1( "h_ele_vertexPhi","ele  momentum #phi",nbinphi,phimin,phimax,"#phi (rad)");
  h_ele_vertexX = bookH1( "h_ele_vertexX","ele vertex x",nbinxyz,-0.1,0.1,"x (cm)" );
  h_ele_vertexY = bookH1( "h_ele_vertexY","ele vertex y",nbinxyz,-0.1,0.1,"y (cm)" );
  h_ele_vertexZ = bookH1( "h_ele_vertexZ","ele vertex z",nbinxyz,-25, 25,"z (cm)" );
//  h_ele_vertexP = bookH1("h_ele_vertexP",        "ele p at vertex",       nbinp,0.,pmax,"p_{vertex} (GeV/c)");
//  h_ele_Et = bookH1( "h_ele_Et","ele SC transverse energy",  nbinpt,0.,ptmax,"E_{T} (GeV)");
//  h_ele_vertexTIP = bookH1( "h_ele_vertexTIP","ele transverse impact parameter (wrt bs)",90,0.,0.15,"TIP (cm)");
//  h_ele_charge = bookH1( "h_ele_charge","ele charge",5,-2.,2.,"charge");

  // # rec electrons
  histNum_= bookH1("h_recEleNum","# rec electrons",20, 0.,20.,"N_{ele}");

  // SuperClusters
  histSclEn_ = bookH1("h_scl_energy","ele supercluster energy",nbinp,0.,pmax);
  histSclEt_ = bookH1("h_scl_et","ele supercluster transverse energy",nbinpt,0.,ptmax);
  histSclEta_ = bookH1("h_scl_eta","ele supercluster eta",nbineta,etamin,etamax);
  histSclPhi_ = bookH1("h_scl_phi","ele supercluster phi",nbinphi,phimin,phimax);
  histSclSigEtaEta_ = bookH1("h_scl_sigetaeta","ele supercluster sigma eta eta",100,0.,0.05);

  // electron track
//  h_ele_ambiguousTracks = bookH1("h_ele_ambiguousTracks","ele # ambiguous tracks",  5,0.,5.);
//  h_ele_ambiguousTracksVsEta = bookH2("h_ele_ambiguousTracksVsEta","ele # ambiguous tracks  vs eta",  nbineta2D,etamin,etamax,5,0.,5.);
//  h_ele_ambiguousTracksVsPhi = bookH2("h_ele_ambiguousTracksVsPhi","ele # ambiguous tracks  vs phi",  nbinphi2D,phimin,phimax,5,0.,5.);
//  h_ele_ambiguousTracksVsPt = bookH2("h_ele_ambiguousTracksVsPt","ele # ambiguous tracks vs pt",  nbinpt2D,0.,ptmax,5,0.,5.);
  h_ele_foundHits = bookH1("h_ele_foundHits","ele track # found hits",nbinfhits,0.,fhitsmax,"N_{hits}");
  h_ele_foundHitsVsEta = bookH2("h_ele_foundHitsVsEta","ele track # found hits vs eta",  nbineta2D,etamin,etamax,nbinfhits,0.,fhitsmax);
  h_ele_foundHitsVsPhi = bookH2("h_ele_foundHitsVsPhi","ele track # found hits vs phi",  nbinphi2D,phimin,phimax,nbinfhits,0.,fhitsmax);
  h_ele_foundHitsVsPt = bookH2("h_ele_foundHitsVsPt","ele track # found hits vs pt",  nbinpt2D,0.,ptmax,nbinfhits,0.,fhitsmax);
  h_ele_lostHits = bookH1("h_ele_lostHits","ele track # lost hits",5,0.,5.,"N_{lost hits}");
  h_ele_lostHitsVsEta = bookH2("h_ele_lostHitsVsEta","ele track # lost hits vs eta",nbineta2D,etamin,etamax,nbinlhits,0.,lhitsmax);
  h_ele_lostHitsVsPhi = bookH2("h_ele_lostHitsVsPhi","ele track # lost hits vs eta",nbinphi2D,phimin,phimax,nbinlhits,0.,lhitsmax);
  h_ele_lostHitsVsPt = bookH2("h_ele_lostHitsVsPt","ele track # lost hits vs eta",nbinpt2D,0.,ptmax,nbinlhits,0.,lhitsmax);
  h_ele_chi2 = bookH1("h_ele_chi2","ele track #chi^{2}",100,0.,15.,"#Chi^{2}");
  h_ele_chi2VsEta = bookH2("h_ele_chi2VsEta","ele track #chi^{2} vs eta",nbineta2D,etamin,etamax,50,0.,15.);
  h_ele_chi2VsPhi = bookH2("h_ele_chi2VsPhi","ele track #chi^{2} vs phi",nbinphi2D,phimin,phimax,50,0.,15.);
  h_ele_chi2VsPt = bookH2("h_ele_chi2VsPt","ele track #chi^{2} vs pt",nbinpt2D,0.,ptmax,50,0.,15.);

  // electron matching and ID
  h_ele_Eop = bookH1( "h_ele_Eop","ele E/P_{vertex}",nbineop,0.,eopmax,"E/P_{vertex}");
  h_ele_EopVsEta = bookH2("h_ele_EopVsEta","ele E/P_{vertex} vs eta",nbineta2D,etamin,etamax,nbineop,0.,eopmax,"E/P_{vertex}");
  h_ele_EopVsPhi = bookH2("h_ele_EopVsPhi","ele E/P_{vertex} vs phi",nbinphi2D,phimin,phimax,nbineop,0.,eopmax,"E/P_{vertex}");
  h_ele_EopVsPt = bookH2("h_ele_EopVsPt","ele E/P_{vertex} vs pt",nbinpt2D,0.,ptmax,nbineop,0.,eopmax,"E/P_{vertex}");
  //h_ele_EoPout = bookH1( "h_ele_EoPout","ele E/P_{out}",nbineop,0.,eopmax,"E_{seed}/P_{out}");
  h_ele_EeleOPout = bookH1( "h_ele_EeleOPout","ele E_{ele}/P_{out}",nbineop,0.,eopmax,"E_{ele}/P_{out}");
  h_ele_EeleOPoutVsEta = bookH2("h_ele_EeleOPoutVsEta","ele E_{ele}/P_{out} vs eta",nbineta2D,etamin,etamax,nbineop,0.,eopmax,"E_{ele}/P_{out}");
  h_ele_EeleOPoutVsPhi = bookH2("h_ele_EeleOPoutVsPhi","ele E_{ele}/P_{out} vs phi",nbinphi2D,phimin,phimax,nbineop,0.,eopmax,"E_{ele}/P_{out}");
  h_ele_EeleOPoutVsPt = bookH2("h_ele_EeleOPoutVsPt","ele E_{ele}/P_{out} vs pt",nbinpt2D,0.,ptmax,nbineop,0.,eopmax,"E_{ele}/P_{out}");
  h_ele_dEtaSc_propVtx = bookH1( "h_ele_dEtaSc_propVtx","ele #eta_{sc} - #eta_{tr}, prop from vertex",nbindetamatch,detamatchmin,detamatchmax,"#eta_{sc} - #eta_{tr}");
  h_ele_dEtaSc_propVtxVsEta = bookH2("h_ele_dEtaSc_propVtxVsEta","ele #eta_{sc} - #eta_{tr}, prop from vertex vs eta",nbineta2D,etamin,etamax,nbindetamatch,detamatchmin,detamatchmax,"#eta_{sc} - #eta_{tr}");
  h_ele_dEtaSc_propVtxVsPhi = bookH2("h_ele_dEtaSc_propVtxVsPhi","ele #eta_{sc} - #eta_{tr}, prop from vertex vs phi",nbinphi2D,phimin,phimax,nbindetamatch,detamatchmin,detamatchmax,"#eta_{sc} - #eta_{tr}");
  h_ele_dEtaSc_propVtxVsPt = bookH2("h_ele_dEtaSc_propVtxVsPt","ele #eta_{sc} - #eta_{tr}, prop from vertex vs pt",nbinpt2D,0.,ptmax,nbindetamatch,detamatchmin,detamatchmax,"#eta_{sc} - #eta_{tr}");
  h_ele_dPhiSc_propVtx = bookH1( "h_ele_dPhiSc_propVtx","ele #phi_{sc} - #phi_{tr}, prop from vertex",nbindphimatch,dphimatchmin,dphimatchmax,"#phi_{sc} - #phi_{tr} (rad)");
  h_ele_dPhiSc_propVtxVsEta = bookH2("h_ele_dPhiSc_propVtxVsEta","ele #phi_{sc} - #phi_{tr}, prop from vertex vs eta",nbineta2D,etamin,etamax,nbindphimatch,dphimatchmin,dphimatchmax,"#phi_{sc} - #phi_{tr} (rad)");
  h_ele_dPhiSc_propVtxVsPhi = bookH2("h_ele_dPhiSc_propVtxVsPhi","ele #phi_{sc} - #phi_{tr}, prop from vertex vs phi",nbinphi2D,phimin,phimax,nbindphimatch,dphimatchmin,dphimatchmax,"#phi_{sc} - #phi_{tr} (rad)");
  h_ele_dPhiSc_propVtxVsPt = bookH2("h_ele_dPhiSc_propVtxVsPt","ele #phi_{sc} - #phi_{tr}, prop from vertex vs pt",nbinpt2D,0.,ptmax,nbindphimatch,dphimatchmin,dphimatchmax,"#phi_{sc} - #phi_{tr} (rad)");
  //h_ele_dEtaCl_propOut = bookH1( "h_ele_dEtaCl_propOut","ele #eta_{cl} - #eta_{tr}, prop from outermost",nbindetamatch,detamatchmin,detamatchmax,"#eta_{seedcl} - #eta_{tr}");
  //h_ele_dPhiCl_propOut = bookH1( "h_ele_dPhiCl_propOut","ele #phi_{cl} - #phi_{tr}, prop from outermost",nbindphimatch,dphimatchmin,dphimatchmax,"#phi_{seedcl} - #phi_{tr} (rad)");
  h_ele_dEtaEleCl_propOut = bookH1( "h_ele_dEtaEleCl_propOut","ele #eta_{EleCl} - #eta_{tr}, prop from outermost",nbindetamatch,detamatchmin,detamatchmax,"#eta_{elecl} - #eta_{tr}");
  h_ele_dEtaEleCl_propOutVsEta = bookH2("h_ele_dEtaEleCl_propOutVsEta","ele #eta_{EleCl} - #eta_{tr}, prop from outermost vs eta",nbineta2D,etamin,etamax,nbindetamatch,detamatchmin,detamatchmax,"#eta_{elecl} - #eta_{tr}");
  h_ele_dEtaEleCl_propOutVsPhi = bookH2("h_ele_dEtaEleCl_propOutVsPhi","ele #eta_{EleCl} - #eta_{tr}, prop from outermost vs phi",nbinphi2D,phimin,phimax,nbindetamatch,detamatchmin,detamatchmax,"#eta_{elecl} - #eta_{tr}");
  h_ele_dEtaEleCl_propOutVsPt = bookH2("h_ele_dEtaEleCl_propOutVsPt","ele #eta_{EleCl} - #eta_{tr}, prop from outermost vs pt",nbinpt2D,0.,ptmax,nbindetamatch,detamatchmin,detamatchmax,"#eta_{elecl} - #eta_{tr}");
  h_ele_dPhiEleCl_propOut = bookH1( "h_ele_dPhiEleCl_propOut","ele #phi_{EleCl} - #phi_{tr}, prop from outermost",nbindphimatch,dphimatchmin,dphimatchmax,"#phi_{elecl} - #phi_{tr} (rad)");
  h_ele_dPhiEleCl_propOutVsEta = bookH2("h_ele_dPhiEleCl_propOutVsEta","ele #phi_{EleCl} - #phi_{tr}, prop from outermost vs eta",nbineta2D,etamin,etamax,nbindphimatch,dphimatchmin,dphimatchmax,"#phi_{elecl} - #phi_{tr} (rad)");
  h_ele_dPhiEleCl_propOutVsPhi = bookH2("h_ele_dPhiEleCl_propOutVsPhi","ele #phi_{EleCl} - #phi_{tr}, prop from outermost vs phi",nbinphi2D,phimin,phimax,nbindphimatch,dphimatchmin,dphimatchmax,"#phi_{elecl} - #phi_{tr} (rad)");
  h_ele_dPhiEleCl_propOutVsPt = bookH2("h_ele_dPhiEleCl_propOutVsPt","ele #phi_{EleCl} - #phi_{tr}, prop from outermost vs pt",nbinpt2D,0.,ptmax,nbindphimatch,dphimatchmin,dphimatchmax,"#phi_{elecl} - #phi_{tr} (rad)");
  h_ele_Hoe = bookH1("h_ele_Hoe","ele hadronic energy / em energy", nbinhoe, hoemin, hoemax,"H/E") ;
  h_ele_HoeVsEta = bookH2("h_ele_HoeVsEta","ele hadronic energy / em energy vs eta",nbineta2D,etamin,etamax,nbinhoe,hoemin,hoemax,"H/E") ;
  h_ele_HoeVsPhi = bookH2("h_ele_HoeVsPhi","ele hadronic energy / em energy vs phi",nbinphi2D,phimin,phimax,nbinhoe,hoemin,hoemax,"H/E") ;
  h_ele_HoeVsPt = bookH2("h_ele_HoeVsPt","ele hadronic energy / em energy vs pt",nbinpt2D,0.,ptmax,nbinhoe,hoemin,hoemax,"H/E") ;
//  h_ele_outerP = bookH1( "h_ele_outerP","ele track outer p, mean",nbinp,0.,pmax,"P_{out} (GeV/c)");
//  h_ele_outerP_mode = bookH1( "h_ele_outerP_mode","ele track outer p, mode",nbinp,0.,pmax,"P_{out} (GeV/c)");
  h_ele_innerPt_mean = bookH1( "h_ele_innerPt_mean","ele track inner p_{T}, mean",nbinpt,0.,ptmax,"P_{T in} (GeV/c)");
  h_ele_outerPt_mean = bookH1( "h_ele_outerPt_mean","ele track outer p_{T}, mean",nbinpt,0.,ptmax,"P_{T out} (GeV/c)");
  h_ele_outerPt_mode = bookH1( "h_ele_outerPt_mode","ele track outer p_{T}, mode",nbinpt,0.,ptmax,"P_{T out} (GeV/c)");


//  h_ele_PinMnPout_mode      = dbe_->book1D( "h_ele_PinMnPout_mode","ele track inner p - outer p, mode"   ,nbinp,0.,100.);
  h_ele_PinMnPout = bookH1( "h_ele_PinMnPout","ele track inner p - outer p, mean" ,nbinp,0.,200.,"P_{vertex} - P_{out} (GeV/c)");
  h_ele_PinMnPout_mode = bookH1( "h_ele_PinMnPout_mode","ele track inner p - outer p, mode",nbinp,0.,100.,"P_{vertex} - P_{out}, mode (GeV/c)");

  h_ele_fbrem = bookH1("h_ele_fbrem","ele brem fraction",100,0.,1.,"P_{in} - P_{out} / P_{in}") ;
  h_ele_fbremVsEta = bookH2("h_ele_fbremVsEta","ele brem fraction vs eta",nbineta2D,etamin,etamax,100,0.,1.,"P_{in} - P_{out} / P_{in}") ;
  h_ele_fbremVsPhi = bookH2("h_ele_fbremVsPhi","ele brem fraction vs phi",nbinphi2D,phimin,phimax,100,0.,1.,"P_{in} - P_{out} / P_{in}") ;
  h_ele_fbremVsPt = bookH2("h_ele_fbremVsPt","ele brem fraction vs pt",nbinpt2D,0.,ptmax,100,0.,1.,"P_{in} - P_{out} / P_{in}") ;
  h_ele_classes = bookH1( "h_ele_classes", "ele electron classes",      150,0.0,150.);

  h_ele_mva = bookH1( "h_ele_mva","ele identification mva",100,-1.,1.);
  h_ele_provenance = bookH1( "h_ele_provenance","ele provenance",5,-2.,3.);

  // isolation
  h_ele_tkSumPt_dr03 = bookH1("h_ele_tkSumPt_dr03","tk isolation sum, dR=0.3",100,0.0,20.,"TkIsoSum, cone 0.3 (GeV/c)");
  h_ele_ecalRecHitSumEt_dr03 = bookH1("h_ele_ecalRecHitSumEt_dr03","ecal isolation sum, dR=0.3",100,0.0,20.,"EcalIsoSum, cone 0.3 (GeV)");
  h_ele_hcalDepth1TowerSumEt_dr03 = bookH1("h_ele_hcalDepth1TowerSumEt_dr03","hcal depth1 isolation sum, dR=0.3",100,0.0,20.,"Hcal1IsoSum, cone 0.3 (GeV)");
  h_ele_hcalDepth2TowerSumEt_dr03 = bookH1("h_ele_hcalDepth2TowerSumEt_dr03","hcal depth2 isolation sum, dR=0.3",100,0.0,20.,"Hcal2IsoSum, cone 0.3 (GeV)");
  h_ele_tkSumPt_dr04 = bookH1("h_ele_tkSumPt_dr04","hcal isolation sum",100,0.0,20.,"TkIsoSum, cone 0.4 (GeV/c)");
  h_ele_ecalRecHitSumEt_dr04 = bookH1("h_ele_ecalRecHitSumEt_dr04","ecal isolation sum, dR=0.4",100,0.0,20.,"EcalIsoSum, cone 0.4 (GeV)");
  h_ele_hcalDepth1TowerSumEt_dr04 = bookH1("h_ele_hcalDepth1TowerSumEt_dr04","hcal depth1 isolation sum, dR=0.4",100,0.0,20.,"Hcal1IsoSum, cone 0.4 (GeV)");
  h_ele_hcalDepth2TowerSumEt_dr04 = bookH1("h_ele_hcalDepth2TowerSumEt_dr04","hcal depth2 isolation sum, dR=0.4",100,0.0,20.,"Hcal2IsoSum, cone 0.4 (GeV)");

  // T&P
  h_ele_mee_os = bookH1("h_ele_mee_os","ele pairs invariant mass, opposite sign", nbinmee, meemin, meemax,"m_{e^{+}e^{-}} (GeV/c^{2})");
  h_ele_mee = bookH1("h_ele_mee","ele pairs invariant mass", nbinmee, meemin, meemax,"m_{ee} (GeV/c^{2})");


//
//  //===========================
//  // histos for matching and matched matched objects
//  //===========================
//
//  // matching object
//  std::string matchingObjectType ;
//  if (std::string::npos!=matchingObjectCollection_.label().find("SuperCluster",0))
//   { matchingObjectType = "SC" ; }
//  if (matchingObjectType=="")
//   { edm::LogError("ElectronMcFakeValidator::beginJob")<<"Unknown matching object type !" ; }
//  else
//   { edm::LogInfo("ElectronMcFakeValidator::beginJob")<<"Matching object type: "<<matchingObjectType ; }
//  std::string htitle = "# "+matchingObjectType+"s", xtitle = "N_{"+matchingObjectType+"}" ;
//  h_matchingObject_Num = bookH1withSumw2("h_matchingObject_Num",htitle,nbinfhits,0.,fhitsmax,xtitle) ;
//
//  // matching object distributions
//  h_matchingObject_Eta = bookH1("h_matchingObject_Eta",matchingObjectType+" #eta",nbineta,etamin,etamax,"#eta_{SC}");
//  h_matchingObject_AbsEta = bookH1("h_matchingObject_AbsEta",matchingObjectType+" |#eta|",nbineta/2,0.,etamax,"|#eta|_{SC}");
//  h_matchingObject_P = bookH1("h_matchingObject_P",matchingObjectType+" p",nbinp,0.,pmax,"E_{SC} (GeV)");
//  h_matchingObject_Pt = bookH1("h_matchingObject_Pt",matchingObjectType+" pt",nbinpteff,5.,ptmax);
//  h_matchingObject_Phi = bookH1("h_matchingObject_Phi",matchingObjectType+" phi",nbinphi,phimin,phimax);
//  h_matchingObject_Z = bookH1("h_matchingObject_Z",matchingObjectType+" z",nbinxyz,-25,25);
//
//  h_matchedObject_Eta = bookH1withSumw2("h_matchedObject_Eta","Efficiency vs matching SC #eta",nbineta,etamin,etamax);
//  h_matchedObject_AbsEta = bookH1withSumw2("h_matchedObject_AbsEta","Efficiency vs matching SC |#eta|",nbineta/2,0.,2.5);
//  h_matchedObject_Pt = bookH1withSumw2("h_matchedObject_Pt","Efficiency vs matching SC E_{T}",nbinpteff,5.,ptmax);
//  h_matchedObject_Phi = bookH1withSumw2("h_matchedObject_Phi","Efficiency vs matching SC phi",nbinphi,phimin,phimax);
//  h_matchedObject_Z = bookH1withSumw2("h_matchedObject_Z","Efficiency vs matching SC z",nbinxyz,-25,25);
//
//  // classes
//  h_matchedEle_eta = bookH1( "h_matchedEle_eta", "ele electron eta",  nbineta/2,0.0,etamax);
//  h_matchedEle_eta_golden = bookH1( "h_matchedEle_eta_golden", "ele electron eta golden",  nbineta/2,0.0,etamax);
//  h_matchedEle_eta_shower = bookH1( "h_matchedEle_eta_shower", "ele electron eta showering",  nbineta/2,0.0,etamax);
//  //h_matchedEle_eta_bbrem = bookH1( "h_matchedEle_eta_bbrem", "ele electron eta bbrem",  nbineta/2,0.0,etamax);
//  //h_matchedEle_eta_narrow = bookH1( "h_matchedEle_eta_narrow", "ele electron eta narrow",  nbineta/2,0.0,etamax);
//
 }

void ElectronAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup & iSetup )
{
  nEvents_++ ;
  if (!trigger(iEvent)) return ;
  nAfterTrigger_++ ;

//  edm::Handle<SuperClusterCollection> barrelSCs ;
//  iEvent.getByLabel("correctedHybridSuperClusters",barrelSCs) ;
//  edm::Handle<SuperClusterCollection> endcapsSCs ;
//  iEvent.getByLabel("correctedMulti5x5SuperClustersWithPreshower",endcapsSCs) ;
//  std::cout<<"[ElectronMcSignalValidator::analyze]"
//    <<"Event "<<iEvent.id()
//    <<" has "<<barrelSCs.product()->size()<<" barrel superclusters"
//    <<" and "<<endcapsSCs.product()->size()<<" endcaps superclusters" ;
//
  edm::Handle<GsfElectronCollection> gsfElectrons ;
  iEvent.getByLabel(electronCollection_,gsfElectrons) ;
  edm::Handle<reco::SuperClusterCollection> recoClusters ;
  iEvent.getByLabel(matchingObjectCollection_,recoClusters) ;
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle ;
  iEvent.getByType(recoBeamSpotHandle) ;
  const BeamSpot bs = *recoBeamSpotHandle ;

  edm::LogInfo("ElectronMcSignalValidator::analyze")
    <<"Treating event "<<iEvent.id()
    <<" with "<<gsfElectrons.product()->size()<<" electrons" ;
  histNum_->Fill((*gsfElectrons).size()) ;

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

    // electron related distributions
    h_ele_vertexPt->Fill( gsfIter->pt() );
    h_ele_vertexEta->Fill( gsfIter->eta() );
    h_ele_vertexPhi->Fill( gsfIter->phi() );
    h_ele_vertexX->Fill( gsfIter->vertex().x() );
    h_ele_vertexY->Fill( gsfIter->vertex().y() );
    h_ele_vertexZ->Fill( gsfIter->vertex().z() );
//    h_ele_vertexP->Fill( gsfIter->p() );
//    h_ele_Et->Fill( gsfIter->superCluster()->energy()/cosh( gsfIter->superCluster()->eta()) );
//    h_ele_vertexTIP->Fill( vertexTIP );
//    h_ele_charge->Fill( gsfIter->charge() );

    // supercluster related distributions
    reco::SuperClusterRef sclRef = gsfIter->superCluster() ;
    // ALREADY DONE IN GSF ELECTRON CORE
    //    if (!gsfIter->ecalDrivenSeed()&&gsfIter->trackerDrivenSeed())
    //      sclRef = gsfIter->pflowSuperCluster() ;
    histSclEn_->Fill(sclRef->energy());
    double R=TMath::Sqrt(sclRef->x()*sclRef->x() + sclRef->y()*sclRef->y() +sclRef->z()*sclRef->z());
    double Rt=TMath::Sqrt(sclRef->x()*sclRef->x() + sclRef->y()*sclRef->y());
    histSclEt_->Fill(sclRef->energy()*(Rt/R));
    histSclEta_->Fill(sclRef->eta());
    histSclPhi_->Fill(sclRef->phi());
    histSclSigEtaEta_->Fill(gsfIter->scSigmaEtaEta());

    // track related distributions
//    h_ele_ambiguousTracks->Fill( gsfIter->ambiguousGsfTracksSize() );
//    h_ele_ambiguousTracksVsEta->Fill( gsfIter->eta(), gsfIter->ambiguousGsfTracksSize() );
//    h_ele_ambiguousTracksVsPhi->Fill( gsfIter->phi(), gsfIter->ambiguousGsfTracksSize() );
//    h_ele_ambiguousTracksVsPt->Fill( gsfIter->pt(), gsfIter->ambiguousGsfTracksSize() );
    if (!readAOD_)
     { // track extra does not exist in AOD
      h_ele_foundHits->Fill( gsfIter->gsfTrack()->numberOfValidHits() );
      h_ele_foundHitsVsEta->Fill( gsfIter->eta(), gsfIter->gsfTrack()->numberOfValidHits() );
      h_ele_foundHitsVsPhi->Fill( gsfIter->phi(), gsfIter->gsfTrack()->numberOfValidHits() );
      h_ele_foundHitsVsPt->Fill( gsfIter->pt(), gsfIter->gsfTrack()->numberOfValidHits() );
      h_ele_lostHits->Fill( gsfIter->gsfTrack()->numberOfLostHits() );
      h_ele_lostHitsVsEta->Fill( gsfIter->eta(), gsfIter->gsfTrack()->numberOfLostHits() );
      h_ele_lostHitsVsPhi->Fill( gsfIter->phi(), gsfIter->gsfTrack()->numberOfLostHits() );
      h_ele_lostHitsVsPt->Fill( gsfIter->pt(), gsfIter->gsfTrack()->numberOfLostHits() );
      h_ele_chi2->Fill( gsfIter->gsfTrack()->normalizedChi2() );
      h_ele_chi2VsEta->Fill( gsfIter->eta(), gsfIter->gsfTrack()->normalizedChi2() );
      h_ele_chi2VsPhi->Fill( gsfIter->phi(), gsfIter->gsfTrack()->normalizedChi2() );
      h_ele_chi2VsPt->Fill( gsfIter->pt(), gsfIter->gsfTrack()->normalizedChi2() );
     }

    // from gsf track interface, hence using mean
    if (!readAOD_)
     { // track extra does not exist in AOD
      h_ele_PinMnPout->Fill( gsfIter->gsfTrack()->innerMomentum().R() - gsfIter->gsfTrack()->outerMomentum().R() );
      //h_ele_outerP->Fill( gsfIter->gsfTrack()->outerMomentum().R() );
      h_ele_innerPt_mean->Fill( gsfIter->gsfTrack()->innerMomentum().Rho() );
      h_ele_outerPt_mean->Fill( gsfIter->gsfTrack()->outerMomentum().Rho() );
     }

    // from electron interface, hence using mode
    h_ele_PinMnPout_mode->Fill( gsfIter->trackMomentumAtVtx().R() - gsfIter->trackMomentumOut().R() );
    //h_ele_outerP_mode->Fill( gsfIter->trackMomentumOut().R() );
    h_ele_outerPt_mode->Fill( gsfIter->trackMomentumOut().Rho() );

    // match distributions
    h_ele_Eop->Fill( gsfIter->eSuperClusterOverP() );
    h_ele_EopVsEta->Fill( gsfIter->eta(), gsfIter->eSuperClusterOverP() );
    h_ele_EopVsPhi->Fill( gsfIter->phi(), gsfIter->eSuperClusterOverP() );
    h_ele_EopVsPt->Fill( gsfIter->pt(), gsfIter->eSuperClusterOverP() );
    h_ele_EeleOPout->Fill( gsfIter->eEleClusterOverPout() );
    h_ele_EeleOPoutVsEta->Fill( gsfIter->eta(), gsfIter->eEleClusterOverPout() );
    h_ele_EeleOPoutVsPhi->Fill( gsfIter->phi(), gsfIter->eEleClusterOverPout() );
    h_ele_EeleOPoutVsPt->Fill( gsfIter->pt(), gsfIter->eEleClusterOverPout() );
    h_ele_dEtaSc_propVtx->Fill(gsfIter->deltaEtaSuperClusterTrackAtVtx());
    h_ele_dEtaSc_propVtxVsEta->Fill(gsfIter->eta(), gsfIter->deltaEtaSuperClusterTrackAtVtx());
    h_ele_dEtaSc_propVtxVsPhi->Fill(gsfIter->phi(), gsfIter->deltaEtaSuperClusterTrackAtVtx());
    h_ele_dEtaSc_propVtxVsPt->Fill(gsfIter->pt(), gsfIter->deltaEtaSuperClusterTrackAtVtx());
    h_ele_dPhiSc_propVtx->Fill(gsfIter->deltaPhiSuperClusterTrackAtVtx());
    h_ele_dPhiSc_propVtxVsEta->Fill(gsfIter->eta(), gsfIter->deltaPhiSuperClusterTrackAtVtx());
    h_ele_dPhiSc_propVtxVsPhi->Fill(gsfIter->phi(), gsfIter->deltaPhiSuperClusterTrackAtVtx());
    h_ele_dPhiSc_propVtxVsPt->Fill(gsfIter->pt(), gsfIter->deltaPhiSuperClusterTrackAtVtx());
    h_ele_dEtaEleCl_propOut->Fill(gsfIter->deltaEtaEleClusterTrackAtCalo());
    h_ele_dEtaEleCl_propOutVsEta->Fill(gsfIter->eta(), gsfIter->deltaEtaEleClusterTrackAtCalo());
    h_ele_dEtaEleCl_propOutVsPhi->Fill(gsfIter->phi(), gsfIter->deltaEtaEleClusterTrackAtCalo());
    h_ele_dEtaEleCl_propOutVsPt->Fill(gsfIter->pt(), gsfIter->deltaEtaEleClusterTrackAtCalo());
    h_ele_dPhiEleCl_propOut->Fill(gsfIter->deltaPhiEleClusterTrackAtCalo());
    h_ele_dPhiEleCl_propOutVsEta->Fill(gsfIter->eta(), gsfIter->deltaPhiEleClusterTrackAtCalo());
    h_ele_dPhiEleCl_propOutVsPhi->Fill(gsfIter->phi(), gsfIter->deltaPhiEleClusterTrackAtCalo());
    h_ele_dPhiEleCl_propOutVsPt->Fill(gsfIter->pt(), gsfIter->deltaPhiEleClusterTrackAtCalo());
    h_ele_Hoe->Fill(gsfIter->hadronicOverEm());
    h_ele_HoeVsEta->Fill(gsfIter->eta(), gsfIter->hadronicOverEm());
    h_ele_HoeVsPhi->Fill(gsfIter->phi(), gsfIter->hadronicOverEm());
    h_ele_HoeVsPt->Fill(gsfIter->pt(), gsfIter->hadronicOverEm());

    h_ele_fbrem->Fill(gsfIter->fbrem()) ;
    h_ele_fbremVsEta->Fill(gsfIter->eta(),gsfIter->fbrem()) ;
    h_ele_fbremVsPhi->Fill(gsfIter->phi(),gsfIter->fbrem()) ;
    h_ele_fbremVsPt->Fill(gsfIter->pt(),gsfIter->fbrem()) ;

    h_ele_mva->Fill(gsfIter->mva()) ;

    if (gsfIter->ecalDrivenSeed()) h_ele_provenance->Fill(1.) ;
    if (gsfIter->trackerDrivenSeed()) h_ele_provenance->Fill(-1.) ;
    if (gsfIter->trackerDrivenSeed()||gsfIter->ecalDrivenSeed()) h_ele_provenance->Fill(0.);
    if (gsfIter->trackerDrivenSeed()&&!gsfIter->ecalDrivenSeed()) h_ele_provenance->Fill(-2.);
    if (!gsfIter->trackerDrivenSeed()&&gsfIter->ecalDrivenSeed()) h_ele_provenance->Fill(2.);

    h_ele_tkSumPt_dr03->Fill(gsfIter->dr03TkSumPt());
    h_ele_ecalRecHitSumEt_dr03->Fill(gsfIter->dr03EcalRecHitSumEt());
    h_ele_hcalDepth1TowerSumEt_dr03->Fill(gsfIter->dr03HcalDepth1TowerSumEt());
    h_ele_hcalDepth2TowerSumEt_dr03->Fill(gsfIter->dr03HcalDepth2TowerSumEt());
    h_ele_tkSumPt_dr04->Fill(gsfIter->dr04TkSumPt());
    h_ele_ecalRecHitSumEt_dr04->Fill(gsfIter->dr04EcalRecHitSumEt());
    h_ele_hcalDepth1TowerSumEt_dr04->Fill(gsfIter->dr04HcalDepth1TowerSumEt());
    h_ele_hcalDepth2TowerSumEt_dr04->Fill(gsfIter->dr04HcalDepth2TowerSumEt());

   }

  // association matching object-reco electrons
//  int matchingObjectNum=0;
  reco::SuperClusterCollection::const_iterator moIter ;
  for
   ( moIter=recoClusters->begin() ;
     moIter!=recoClusters->end() ;
     moIter++ )
   {
//    // number of matching objects
//    matchingObjectNum++;

    if
     ( moIter->energy()/cosh(moIter->eta())>maxPtMatchingObject_ ||
       fabs(moIter->eta())> maxAbsEtaMatchingObject_ )
     { continue ; }

//    // suppress the endcaps
//    //if (fabs(moIter->eta()) > 1.5) continue;
//    // select central z
//    //if ( fabs((*mcIter)->production_vertex()->position().z())>50.) continue;
//
//    h_matchingObject_Eta->Fill( moIter->eta() );
//    h_matchingObject_AbsEta->Fill( fabs(moIter->eta()) );
//    h_matchingObject_P->Fill( moIter->energy() );
//    h_matchingObject_Pt->Fill( moIter->energy()/cosh(moIter->eta()) );
//    h_matchingObject_Phi->Fill( moIter->phi() );
//    h_matchingObject_Z->Fill(  moIter->z() );

    if (Selection_<4)
     {
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
          h_ele_mee->Fill(invMass) ;
          if (((gsfIter->charge())*(gsfIter2->charge()))<0.)
           { h_ele_mee_os->Fill(invMass) ; }
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
          if (fabs(dphi)>CLHEP::pi)
           { dphi = dphi < 0? (CLHEP::twopi) + dphi : dphi - CLHEP::twopi ; }
          double deltaR = sqrt(pow((moIter->eta()-gsfIter->eta()),2) + pow(dphi,2)) ;
          if ( deltaR < deltaR_ )
           {
            //if ( (genPc->pdg_id() == 11) && (gsfIter->charge() < 0.) || (genPc->pdg_id() == -11) &&
            //(gsfIter->charge() > 0.) ){
            double tmpGsfRatio = gsfIter->p()/moIter->energy() ;
            if ( fabs(tmpGsfRatio-1) < fabs(gsfOkRatio-1) )
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
       { fillMatchedHistos(moIter,bestGsfElectron) ; }
     }
    else
     {
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

        if (Selection_ >= 4)
         {
          reco::GsfElectronCollection::const_iterator gsfIter2 ;
          for
           ( gsfIter2=gsfIter+1 ;
             gsfIter2!=gsfElectrons->end() ;
             gsfIter2++ )
           {
            float invMass = computeInvMass(*gsfIter,*gsfIter2) ;
            h_ele_mee->Fill(invMass) ;

            if (TPchecksign_ && (((gsfIter->charge())*(gsfIter2->charge()))>=0.)) break ;

            // conditions Tag
            if(TAGcheckclass_ && (gsfIter->classification()==GsfElectron::SHOWERING || gsfIter->isGap())) break;

            // conditions Probe
            if(PROBEetcut_ && (gsfIter2->superCluster()->energy()/cosh(gsfIter2->superCluster()->eta())<minEt_)) continue;
            if(PROBEcheckclass_ && (gsfIter2->classification()==GsfElectron::SHOWERING || gsfIter2->isGap())) continue;

            if( invMass < massLow_ || invMass > massHigh_ ) continue ;

            h_ele_mee_os->Fill(invMass) ;

//            fillMatchedHistos(moIter,*gsfIter2) ;
           }
         }
       }
     } // end of Selection_>=4

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

void ElectronAnalyzer::fillMatchedHistos
 ( const reco::SuperClusterCollection::const_iterator & moIter,
   const reco::GsfElectron & electron )
 {
  // generated distributions for matched electrons
//  h_matchedObject_Eta->Fill( moIter->eta() );
//  h_matchedObject_AbsEta->Fill( fabs(moIter->eta()) );
//  h_matchedObject_Pt->Fill( moIter->energy()/cosh(moIter->eta()) );
//  h_matchedObject_Phi->Fill( moIter->phi() );
//  h_matchedObject_Z->Fill( moIter->z() );

  //classes
//  int eleClass = electron.classification() ;
//  h_ele_classes->Fill(eleClass) ;
//  h_matchedEle_eta->Fill(fabs(electron.eta()));
//  if (electron.classification() == GsfElectron::GOLDEN) h_matchedEle_eta_golden->Fill(fabs(electron.eta()));
//  if (electron.classification() == GsfElectron::SHOWERING) h_matchedEle_eta_shower->Fill(fabs(electron.eta()));
//  //if (electron.classification() == GsfElectron::BIGBREM) h_matchedEle_eta_bbrem->Fill(fabs(electron.eta()));
//  //if (electron.classification() == GsfElectron::OLDNARROW) h_matchedEle_eta_narrow->Fill(fabs(electron.eta()));
 }

bool ElectronAnalyzer::trigger( const edm::Event & e )
 {
  // retreive TriggerResults from the event
  edm::Handle<edm::TriggerResults> triggerResults ;
  e.getByLabel(triggerResults_,triggerResults) ;

  bool accept = false ;

  if (triggerResults.isValid())
   {
    //std::cout << "TriggerResults found, number of HLT paths: " << triggerResults->size() << std::endl;

    // get trigger names
    edm::TriggerNames triggerNames_;
    triggerNames_.init(*triggerResults) ;
    if (nEvents_==1)
     {
      for (unsigned int i=0; i<triggerNames_.size(); i++)
       {
//        std::cout << "trigger path= " << triggerNames_.triggerName(i) << std::endl;
       }
     }

    unsigned int n = HLTPathsByName_.size() ;
    for (unsigned int i=0; i!=n; i++)
     {
      HLTPathsByIndex_[i]=triggerNames_.triggerIndex(HLTPathsByName_[i]) ;
     }

    // empty input vectors (n==0) means any trigger paths
    if (n==0)
     {
      n=triggerResults->size() ;
      HLTPathsByName_.resize(n) ;
      HLTPathsByIndex_.resize(n) ;
      for ( unsigned int i=0 ; i!=n ; i++)
       {
        HLTPathsByName_[i]=triggerNames_.triggerName(i) ;
        HLTPathsByIndex_[i]=i ;
       }
     }

//    if (nEvents_==1)
//     {
//      if (n>0)
//       {
//        std::cout << "HLT trigger paths requested: index, name and valididty:" << std::endl;
//        for (unsigned int i=0; i!=n; i++)
//         {
//          bool validity = HLTPathsByIndex_[i]<triggerResults->size();
//          std::cout
//            << " " << HLTPathsByIndex_[i]
//            << " " << HLTPathsByName_[i]
//            << " " << validity << std::endl;
//         }
//       }
//     }

    // count number of requested HLT paths which have fired
    unsigned int fired=0 ;
    for ( unsigned int i=0 ; i!=n ; i++ )
     {
      if (HLTPathsByIndex_[i]<triggerResults->size())
       {
        if (triggerResults->accept(HLTPathsByIndex_[i]))
         {
          fired++ ;
          //std::cout << "Fired HLT path= " << HLTPathsByName_[i] << std::endl ;
          accept = true ;
         }
       }
     }

   }

  return accept ;
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
  if (fabs(gsfIter->eta())>maxAbsEta_) return true ;
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
  if (gsfIter->isEB() && fabs(gsfIter->deltaEtaSuperClusterTrackAtVtx()) < dEtaMinBarrel_) return true ;
  if (gsfIter->isEB() && fabs(gsfIter->deltaEtaSuperClusterTrackAtVtx()) > dEtaMaxBarrel_) return true ;
  if (gsfIter->isEE() && fabs(gsfIter->deltaEtaSuperClusterTrackAtVtx()) < dEtaMinEndcaps_) return true ;
  if (gsfIter->isEE() && fabs(gsfIter->deltaEtaSuperClusterTrackAtVtx()) > dEtaMaxEndcaps_) return true ;
  if (gsfIter->isEB() && fabs(gsfIter->deltaPhiSuperClusterTrackAtVtx()) < dPhiMinBarrel_) return true ;
  if (gsfIter->isEB() && fabs(gsfIter->deltaPhiSuperClusterTrackAtVtx()) > dPhiMaxBarrel_) return true ;
  if (gsfIter->isEE() && fabs(gsfIter->deltaPhiSuperClusterTrackAtVtx()) < dPhiMinEndcaps_) return true ;
  if (gsfIter->isEE() && fabs(gsfIter->deltaPhiSuperClusterTrackAtVtx()) > dPhiMaxEndcaps_) return true ;
  if (gsfIter->isEB() && gsfIter->scSigmaIEtaIEta() < sigIetaIetaMinBarrel_) return true ;
  if (gsfIter->isEB() && gsfIter->scSigmaIEtaIEta() > sigIetaIetaMaxBarrel_) return true ;
  if (gsfIter->isEE() && gsfIter->scSigmaIEtaIEta() < sigIetaIetaMinEndcaps_) return true ;
  if (gsfIter->isEE() && gsfIter->scSigmaIEtaIEta() > sigIetaIetaMaxEndcaps_) return true ;
  if (gsfIter->isEB() && gsfIter->hadronicOverEm() > hadronicOverEmMaxBarrel_) return true ;
  if (gsfIter->isEE() && gsfIter->hadronicOverEm() > hadronicOverEmMaxEndcaps_) return true ;

  return false ;
 }
