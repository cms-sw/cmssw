
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
  std::string internalPath("Egamma/ElectronAnalyzer/") ;
  if (Selection_==0) { internalPath = "Egamma/ElectronAnalyzer/AllElectrons/" ; }
  else if (Selection_==1) { internalPath = "Egamma/ElectronAnalyzer/SelectionEt/" ; }
  else if (Selection_==2) { internalPath = "Egamma/ElectronAnalyzer/SelectionEtIso/" ; }
  else if (Selection_==3) { internalPath = "Egamma/ElectronAnalyzer/SelectionEtIsoElID/" ; }
  else if (Selection_==4) { internalPath = "Egamma/ElectronAnalyzer/TagAndProbe/" ; }
  else { edm::LogError("ElectronAnalyzer::book")<<"Unknown selection strategy "<<Selection_ ; }
  setStoreFolder(internalPath) ;

  MonitorElement * h_matchedEle_eta = get(internalPath+"h_matchedEle_eta");
  MonitorElement * h_matchedEle_eta_golden = get(internalPath+"h_matchedEle_eta_golden");
  MonitorElement * h_matchedEle_eta_shower = get(internalPath+"h_matchedEle_eta_shower");
  //MonitorElement * h_matchedEle_eta_bbrem = get(internalPath+"h_matchedEle_eta_bbrem");
  //MonitorElement * h_matchedEle_eta_narrow = get(internalPath+"h_matchedEle_eta_narrow");
  MonitorElement * h_matchedEle_eta_goldenFrac = cloneH1("h_matchedEle_eta_goldenFrac",internalPath+"h_matchedEle_eta","fraction of golden electrons") ;
  MonitorElement * h_matchedEle_eta_showerFrac = cloneH1("h_matchedEle_eta_showerFrac",internalPath+"h_matchedEle_eta","fraction of showering electrons") ;
  //MonitorElement * h_matchedEle_eta_bbremFrac = cloneH1("h_matchedEle_eta_bbremFrac",internalPath+"h_matchedEle_eta","fraction of bbrem electrons") ;
  //MonitorElement * h_matchedEle_eta_narrowFrac = cloneH1("h_matchedEle_eta_narrowFrac",internalPath+"h_matchedEle_eta","fraction of narrow electrons") ;
  int nb, nbins=h_matchedEle_eta->getNbinsX() ;
  for (  nb=0 ; nb<nbins ; ++nb )
   {
    float content = h_matchedEle_eta->getBinContent(nb) ;
    if (content==0.) continue ;
    float contgold =( h_matchedEle_eta_golden->getBinContent(nb))/content ;
    float contshower =( h_matchedEle_eta_shower->getBinContent(nb))/content ;
    //float contbbrem =( h_matchedEle_eta_bbrem->getBinContent(nb))/content ;
    //float contnarrow =( h_matchedEle_eta_narrow->getBinContent(nb))/content ;
    h_matchedEle_eta_goldenFrac ->setBinContent(nb,contgold) ;
    h_matchedEle_eta_showerFrac ->setBinContent(nb,contshower) ;
    //h_matchedEle_eta_bbremFrac ->setBinContent(nb,contbbrem) ;
    //h_matchedEle_eta_narrowFrac ->setBinContent(nb,contnarrow) ;
   }
  remove("h_matchedEle_eta") ;
  remove("h_matchedEle_eta_golden") ;
  remove("h_matchedEle_eta_shower") ;
  //remove("h_matchedEle_eta_bbrem") ;
  //remove("h_matchedEle_eta_narrow") ;

  bookH1andDivide("h_ele_ptEff",internalPath+"h_matchedObject_Pt",internalPath+"h_matchingObject_Pt","p_{T} (GeV/c)","Efficiency","fraction of reco ele matching a reco SC") ;
  bookH1andDivide("h_ele_etaEff",internalPath+"h_matchedObject_Eta",internalPath+"h_matchingObject_Eta","#eta","Efficiency","fraction of reco ele matching a reco SC") ;
  bookH1andDivide("h_ele_absEtaEff",internalPath+"h_matchedObject_AbsEta",internalPath+"h_matchingObject_AbsEta","|#eta|","Efficiency","fraction of reco ele matching a reco SC") ;
  bookH1andDivide("h_ele_phiEff",internalPath+"h_matchedObject_Phi",internalPath+"h_matchingObject_Phi","#phi (rad)","Efficiency","fraction of reco ele matching a reco SC") ;
  bookH1andDivide("h_ele_zEff",internalPath+"h_matchedObject_Z",internalPath+"h_matchingObject_Z","cm","Efficiency","fraction of reco ele matching a reco SC") ;
 }

