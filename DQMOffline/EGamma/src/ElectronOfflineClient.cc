
#include "DQMOffline/EGamma/interface/ElectronOfflineClient.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>

ElectronOfflineClient::ElectronOfflineClient( const edm::ParameterSet & conf )
 : ElectronDqmAnalyzerBase(conf)
 {
  Selection_ = conf.getParameter<int>("Selection");
 }

ElectronOfflineClient::~ElectronOfflineClient()
 {}

void ElectronOfflineClient::finalize()
 {
//  MonitorElement * h_matchedEle_eta = get("h_matchedEle_eta");
//  MonitorElement * h_matchedEle_eta_golden = get("h_matchedEle_eta_golden");
//  MonitorElement * h_matchedEle_eta_shower = get("h_matchedEle_eta_shower");
//  //MonitorElement * h_matchedEle_eta_bbrem = get("h_matchedEle_eta_bbrem");
//  //MonitorElement * h_matchedEle_eta_narrow = get("h_matchedEle_eta_narrow");
//  MonitorElement * h_matchedEle_eta_goldenFrac = cloneH1("h_matchedEle_eta_goldenFrac","h_matchedEle_eta","fraction of golden electrons") ;
//  MonitorElement * h_matchedEle_eta_showerFrac = cloneH1("h_matchedEle_eta_showerFrac","h_matchedEle_eta","fraction of showering electrons") ;
//  //MonitorElement * h_matchedEle_eta_bbremFrac = cloneH1("h_matchedEle_eta_bbremFrac","h_matchedEle_eta","fraction of bbrem electrons") ;
//  //MonitorElement * h_matchedEle_eta_narrowFrac = cloneH1("h_matchedEle_eta_narrowFrac","h_matchedEle_eta","fraction of narrow electrons") ;
//  int nb, nbins=h_matchedEle_eta->getNbinsX() ;
//  for (  nb=0 ; nb<nbins ; ++nb )
//   {
//    float content = h_matchedEle_eta->getBinContent(nb) ;
//    if (content==0.) continue ;
//    float contgold =( h_matchedEle_eta_golden->getBinContent(nb))/content ;
//    float contshower =( h_matchedEle_eta_shower->getBinContent(nb))/content ;
//    //float contbbrem =( h_matchedEle_eta_bbrem->getBinContent(nb))/content ;
//    //float contnarrow =( h_matchedEle_eta_narrow->getBinContent(nb))/content ;
//    h_matchedEle_eta_goldenFrac ->setBinContent(nb,contgold) ;
//    h_matchedEle_eta_showerFrac ->setBinContent(nb,contshower) ;
//    //h_matchedEle_eta_bbremFrac ->setBinContent(nb,contbbrem) ;
//    //h_matchedEle_eta_narrowFrac ->setBinContent(nb,contnarrow) ;
//   }
//  remove("h_matchedEle_eta") ;
//  remove("h_matchedEle_eta_golden") ;
//  remove("h_matchedEle_eta_shower") ;
//  //remove("h_matchedEle_eta_bbrem") ;
//  //remove("h_matchedEle_eta_narrow") ;

  bookH1andDivide("h1_ele_ptEff","h1_matchedObject_Pt","h1_matchingObject_Pt","p_{T} (GeV/c)","Efficiency","fraction of reco ele matching a reco SC") ;
  bookH1andDivide("h1_ele_etaEff","h1_matchedObject_Eta","h1_matchingObject_Eta","#eta","Efficiency","fraction of reco ele matching a reco SC") ;
//  bookH1andDivide("h_ele_absEtaEff","h_matchedObject_AbsEta","h_matchingObject_AbsEta","|#eta|","Efficiency","fraction of reco ele matching a reco SC") ;
  bookH1andDivide("h1_ele_phiEff","h1_matchedObject_Phi","h1_matchingObject_Phi","#phi (rad)","Efficiency","fraction of reco ele matching a reco SC") ;
  bookH1andDivide("h1_ele_zEff","h1_matchedObject_Z","h1_matchingObject_Z","cm","Efficiency","fraction of reco ele matching a reco SC") ;
 }

