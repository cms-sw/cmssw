#include "DQMOffline/Trigger/plugins/HTDQM.h"

HTDQM::HTDQM()= default;

HTDQM::~HTDQM()= default;

void HTDQM::initialise(const edm::ParameterSet& iConfig ){

  ht_variable_binning_ = iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("htBinning");
  met_variable_binning_ = iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("metBinning2");
  ht_binning_ = getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("htPSet"));
  ls_binning_ = getHistoPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("htlsPSet"));
  
}

void HTDQM::bookHistograms(DQMStore::IBooker     & ibooker) 
{  
  
  std::string histname, histtitle;

  histname = "ht_variable"; histtitle = "PFHT";
  bookME(ibooker,htME_variableBinning_,histname,histtitle,ht_variable_binning_);
  setMETitle(htME_variableBinning_,"PF HT [GeV]","events / [GeV]");

  histname = "htVsMET"; histtitle = "PFHT vs PFMET";
  bookME(ibooker,htVsMET_,histname,histtitle,met_variable_binning_,ht_variable_binning_);
  setMETitle(htVsMET_,"PF MET [GeV]","PF HT [GeV]");

  histname = "htVsLS"; histtitle = "PFHT vs LS";
  bookME(ibooker,htVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,ht_binning_.xmin, ht_binning_.xmax);
  setMETitle(htVsLS_,"LS","PF HT [GeV]");

}

void HTDQM::fillHistograms(const std::vector<reco::PFJet> & htjets,
			   const double & met,
			   const int & ls,
			   const bool passCond){

  // filling histograms (denominator)  
  double htSum = 0;
  for (auto const & htjet : htjets){
    htSum += htjet.pt(); 
  }

  htME_variableBinning_.denominator -> Fill(htSum);

  htVsMET_.denominator -> Fill(met, htSum);
  htVsLS_.denominator -> Fill(ls, htSum);
  
  // applying selection for numerator
  if (passCond){
    // filling histograms (num_genTriggerEventFlag_)  
    htME_variableBinning_.numerator -> Fill(htSum);
    htVsMET_.numerator -> Fill(met, htSum);
    htVsLS_.numerator -> Fill(ls, htSum);
  }

}

void HTDQM::fillHtDescription(edm::ParameterSetDescription & histoPSet){

  edm::ParameterSetDescription htPSet;
  fillHistoPSetDescription(htPSet);
  histoPSet.add<edm::ParameterSetDescription>("htPSet", htPSet);

  std::vector<double> bins = {0.,50.,100.,150.,200.,250.,300.,350.,400.,450.,500.,550.,600.,650.,700.,750.,800.,900.,1000.,1200.,1500.,2000.};
  histoPSet.add<std::vector<double> >("htBinning", bins);

  std::vector<double> metbins = {0.,20.,40.,60.,80.,100.,120.,140.,160.,180.,200.,220.,240.,260.,280.,300.,320.,340.,360.,380.,400.,450.,500.,1000.};
  histoPSet.add<std::vector<double> >("metBinning2", metbins);

  edm::ParameterSetDescription lsPSet;
  fillHistoLSPSetDescription(lsPSet);
  histoPSet.add<edm::ParameterSetDescription>("htlsPSet", lsPSet);

}
