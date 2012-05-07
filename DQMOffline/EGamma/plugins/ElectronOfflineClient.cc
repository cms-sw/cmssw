
#include "DQMOffline/EGamma/plugins/ElectronOfflineClient.h"
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
//  MonitorElement * h1_matchedEle_eta = get("matchedEle_eta");
//  MonitorElement * h1_matchedEle_eta_golden = get("matchedEle_eta_golden");
//  MonitorElement * h1_matchedEle_eta_shower = get("matchedEle_eta_shower");
//  //MonitorElement * h1_matchedEle_eta_bbrem = get("matchedEle_eta_bbrem");
//  //MonitorElement * h1_matchedEle_eta_narrow = get("matchedEle_eta_narrow");
//  MonitorElement * h1_matchedEle_eta_goldenFrac = cloneH1("matchedEle_eta_goldenFrac","matchedEle_eta","fraction of golden electrons") ;
//  MonitorElement * h1_matchedEle_eta_showerFrac = cloneH1("matchedEle_eta_showerFrac","matchedEle_eta","fraction of showering electrons") ;
//  //MonitorElement * h1_matchedEle_eta_bbremFrac = cloneH1("matchedEle_eta_bbremFrac","matchedEle_eta","fraction of bbrem electrons") ;
//  //MonitorElement * h1_matchedEle_eta_narrowFrac = cloneH1("matchedEle_eta_narrowFrac","matchedEle_eta","fraction of narrow electrons") ;
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
//  remove("matchedEle_eta") ;
//  remove("matchedEle_eta_golden") ;
//  remove("matchedEle_eta_shower") ;
//  //remove("matchedEle_eta_bbrem") ;
//  //remove("matchedEle_eta_narrow") ;

  setBookIndex(100) ;
  if (effHistoTitle_=="")
   {
    bookH1andDivide("ptEff","matchedObject_Pt","matchingObject_Pt","p_{T} (GeV/c)","Efficiency","efficiency vs p_{T}") ;
    bookH1andDivide("etaEff","matchedObject_Eta","matchingObject_Eta","#eta","Efficiency","efficiency vs #eta") ;
  //  bookH1andDivide("absEtaEff","matchedObject_AbsEta","matchingObject_AbsEta","|#eta|","Efficiency","efficiency vs |#eta|") ;
    bookH1andDivide("phiEff","matchedObject_Phi","matchingObject_Phi","#phi (rad)","Efficiency","efficiency vs #phi") ;
    bookH1andDivide("zEff","matchedObject_Z","matchingObject_Z","z (cm)","Efficiency","efficiency vs z") ;
   }
  else
   {
    bookH1andDivide("ptEff","matchedObject_Pt","matchingObject_Pt","p_{T} (GeV/c)","Efficiency",effHistoTitle_) ;
    bookH1andDivide("etaEff","matchedObject_Eta","matchingObject_Eta","#eta","Efficiency",effHistoTitle_) ;
  //  bookH1andDivide("absEtaEff","matchedObject_AbsEta","matchingObject_AbsEta","|#eta|","Efficiency",effHistoTitle_) ;
    bookH1andDivide("phiEff","matchedObject_Phi","matchingObject_Phi","#phi (rad)","Efficiency",effHistoTitle_) ;
    bookH1andDivide("zEff","matchedObject_Z","matchingObject_Z","z (cm)","Efficiency",effHistoTitle_) ;
   }

  remove("matchedObject_Pt") ;
  remove("matchedObject_Eta") ;
//  remove("matchedObject_AbsEta") ;
  remove("matchedObject_Phi") ;
  remove("matchedObject_Z") ;

  remove("matchingObject_Pt") ;
  remove("matchingObject_Eta") ;
//  remove("matchingObject_AbsEta") ;
  remove("matchingObject_Phi") ;
  remove("matchingObject_Z") ;
 }

