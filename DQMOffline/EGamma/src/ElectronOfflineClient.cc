
#include "DQMOffline/EGamma/interface/ElectronOfflineClient.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>

ElectronOfflineClient::ElectronOfflineClient( const edm::ParameterSet & conf )
 : ElectronDqmAnalyzerBase(conf)
 {
  effHistoTitle_ = conf.getParameter<std::string>("EffHistoTitle") ;
 }

ElectronOfflineClient::~ElectronOfflineClient()
 {}

void ElectronOfflineClient::finalize()
 {
//  MonitorElement * h1_matchedEle_eta = get("h1_matchedEle_eta");
//  MonitorElement * h1_matchedEle_eta_golden = get("h1_matchedEle_eta_golden");
//  MonitorElement * h1_matchedEle_eta_shower = get("h1_matchedEle_eta_shower");
//  //MonitorElement * h1_matchedEle_eta_bbrem = get("h1_matchedEle_eta_bbrem");
//  //MonitorElement * h1_matchedEle_eta_narrow = get("h1_matchedEle_eta_narrow");
//  MonitorElement * h1_matchedEle_eta_goldenFrac = cloneH1("h1_matchedEle_eta_goldenFrac","h1_matchedEle_eta","fraction of golden electrons") ;
//  MonitorElement * h1_matchedEle_eta_showerFrac = cloneH1("h1_matchedEle_eta_showerFrac","h1_matchedEle_eta","fraction of showering electrons") ;
//  //MonitorElement * h1_matchedEle_eta_bbremFrac = cloneH1("h1_matchedEle_eta_bbremFrac","h1_matchedEle_eta","fraction of bbrem electrons") ;
//  //MonitorElement * h1_matchedEle_eta_narrowFrac = cloneH1("h1_matchedEle_eta_narrowFrac","h1_matchedEle_eta","fraction of narrow electrons") ;
//  int nb, nbins=h_matchedEle_eta->getNbinsX() ;
//  for (  nb=0 ; nb<nbins ; ++nb )
//   {
//    float content = h1_matchedEle_eta->getBinContent(nb) ;
//    if (content==0.) continue ;
//    float contgold =( h1_matchedEle_eta_golden->getBinContent(nb))/content ;
//    float contshower =( h1_matchedEle_eta_shower->getBinContent(nb))/content ;
//    //float contbbrem =( h1_matchedEle_eta_bbrem->getBinContent(nb))/content ;
//    //float contnarrow =( h1_matchedEle_eta_narrow->getBinContent(nb))/content ;
//    h1_matchedEle_eta_goldenFrac ->setBinContent(nb,contgold) ;
//    h1_matchedEle_eta_showerFrac ->setBinContent(nb,contshower) ;
//    //h1_matchedEle_eta_bbremFrac ->setBinContent(nb,contbbrem) ;
//    //h1_matchedEle_eta_narrowFrac ->setBinContent(nb,contnarrow) ;
//   }
//  remove("h1_matchedEle_eta") ;
//  remove("h1_matchedEle_eta_golden") ;
//  remove("h1_matchedEle_eta_shower") ;
//  //remove("h1_matchedEle_eta_bbrem") ;
//  //remove("h1_matchedEle_eta_narrow") ;

  if (effHistoTitle_=="")
   {
    bookH1andDivide("h1_ele_ptEff","h1_matchedObject_Pt","h1_matchingObject_Pt","p_{T} (GeV/c)","Efficiency","efficiency vs p_{T}") ;
    bookH1andDivide("h1_ele_etaEff","h1_matchedObject_Eta","h1_matchingObject_Eta","#eta","Efficiency","efficiency vs #eta") ;
  //  bookH1andDivide("h1_ele_absEtaEff","h1_matchedObject_AbsEta","h1_matchingObject_AbsEta","|#eta|","Efficiency","efficiency vs |#eta|") ;
    bookH1andDivide("h1_ele_phiEff","h1_matchedObject_Phi","h1_matchingObject_Phi","#phi (rad)","Efficiency","efficiency vs #phi") ;
    bookH1andDivide("h1_ele_zEff","h1_matchedObject_Z","h1_matchingObject_Z","z (cm)","Efficiency","efficiency vs z") ;
   }
  else
   {
    bookH1andDivide("h1_ele_ptEff","h1_matchedObject_Pt","h1_matchingObject_Pt","p_{T} (GeV/c)","Efficiency",effHistoTitle_) ;
    bookH1andDivide("h1_ele_etaEff","h1_matchedObject_Eta","h1_matchingObject_Eta","#eta","Efficiency",effHistoTitle_) ;
  //  bookH1andDivide("h1_ele_absEtaEff","h1_matchedObject_AbsEta","h1_matchingObject_AbsEta","|#eta|","Efficiency",effHistoTitle_) ;
    bookH1andDivide("h1_ele_phiEff","h1_matchedObject_Phi","h1_matchingObject_Phi","#phi (rad)","Efficiency",effHistoTitle_) ;
    bookH1andDivide("h1_ele_zEff","h1_matchedObject_Z","h1_matchingObject_Z","z (cm)","Efficiency",effHistoTitle_) ;
   }

  remove("h1_matchedObject_Pt") ;
  remove("h1_matchedObject_Eta") ;
//  remove("h1_matchedObject_AbsEta") ;
  remove("h1_matchedObject_Phi") ;
  remove("h1_matchedObject_Z") ;

  remove("h1_matchingObject_Pt") ;
  remove("h1_matchingObject_Eta") ;
//  remove("h1_matchingObject_AbsEta") ;
  remove("h1_matchingObject_Phi") ;
  remove("h1_matchingObject_Z") ;
 }

