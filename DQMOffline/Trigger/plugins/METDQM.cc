#include "DQMOffline/Trigger/plugins/METDQM.h"

METDQM::METDQM()= default;

METDQM::~METDQM()= default;

void METDQM::initialise(const edm::ParameterSet& iConfig ){

  met_variable_binning_ = iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("metBinning");
  met_binning_ = getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("metPSet"));
  phi_binning_ = getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("phiPSet"));
  ls_binning_ = getHistoPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("lsPSet")     );
  
}

void METDQM::bookHistograms(DQMStore::IBooker     & ibooker) 
{  
  
  std::string histname, histtitle;

  histname = "met"; histtitle = "PFMET";
  bookME(ibooker,metME_,histname,histtitle,met_binning_.nbins,met_binning_.xmin, met_binning_.xmax);
  setMETitle(metME_,"PF MET [GeV]","events / [GeV]");

  histname = "met_variable"; histtitle = "PFMET";
  bookME(ibooker,metME_variableBinning_,histname,histtitle,met_variable_binning_);
  setMETitle(metME_variableBinning_,"PF MET [GeV]","events / [GeV]");

  histname = "metVsLS"; histtitle = "PFMET vs LS";
  bookME(ibooker,metVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,met_binning_.xmin, met_binning_.xmax);
  setMETitle(metVsLS_,"LS","PF MET [GeV]");

  histname = "metPhi"; histtitle = "PFMET phi";
  bookME(ibooker,metPhiME_,histname,histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
  setMETitle(metPhiME_,"PF MET #phi","events / 0.1 rad");

}

void METDQM::fillHistograms(const double & met,
			    const double & phi,
			    const int & ls,
			    const bool passCond){

  // filling histograms (denominator)  
  metME_.denominator -> Fill(met);
  metME_variableBinning_.denominator -> Fill(met);
  metPhiME_.denominator -> Fill(phi);

  metVsLS_.denominator -> Fill(ls, met);
  
  // applying selection for numerator
  if (passCond){
    // filling histograms (num_genTriggerEventFlag_)  
    metME_.numerator -> Fill(met);
    metME_variableBinning_.numerator -> Fill(met);
    metPhiME_.numerator -> Fill(phi);
    metVsLS_.numerator -> Fill(ls, met);
  }

}

void METDQM::fillMetDescription(edm::ParameterSetDescription & histoPSet){

  edm::ParameterSetDescription metPSet;
  fillHistoPSetDescription(metPSet);
  histoPSet.add<edm::ParameterSetDescription>("metPSet", metPSet);

  std::vector<double> bins = {0.,20.,40.,60.,80.,90.,100.,110.,120.,130.,140.,150.,160.,170.,180.,190.,200.,220.,240.,260.,280.,300.,350.,400.,450.,1000.};
  histoPSet.add<std::vector<double> >("metBinning", bins);

  edm::ParameterSetDescription phiPSet;
  fillHistoPSetDescription(phiPSet);
  histoPSet.add<edm::ParameterSetDescription>("phiPSet", phiPSet);

  edm::ParameterSetDescription lsPSet;
  fillHistoLSPSetDescription(lsPSet);
  histoPSet.add<edm::ParameterSetDescription>("lsPSet", lsPSet);

}
