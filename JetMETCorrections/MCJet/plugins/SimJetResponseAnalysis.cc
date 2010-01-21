//   
//  SimJetResponse 
//  Initial version November 22, 2006   Anwar A Bhatti  The Rockefeller University, New York NY

#include <iostream>
#include <sstream>
#include <istream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>
#include <functional>

#include "SimJetResponseAnalysis.h"


#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"


#include "CaloTowerBoundriesMC.h"
#include "JetUtilMC.h"

typedef CaloJetCollection::const_iterator CalJetIter;
typedef GenJetCollection::const_iterator GenJetIter;

void GetMPV(TH1* h,double& mean,double& width,double& error);

SimJetResponseAnalysis::SimJetResponseAnalysis(edm::ParameterSet const& cfg) {

  //  std::cout << " Beginning SimJetResponse Analysis " << std::endl;

  MatchRadius_ = 0.2;
  RecJetPtMin_   = 5.;
  NJetMax_   = 2;

  genjets_    = cfg.getParameter<std::string> ("genjets");
  recjets_    = cfg.getParameter<std::string> ("recjets");
  genmet_     = cfg.getParameter<std::string> ("genmet");
  recmet_     = cfg.getParameter<std::string> ("recmet");

  NJetMax_ = cfg.getParameter<int> ("NJetMax");
  MatchRadius_ =cfg.getParameter<double> ("MatchRadius");
  RecJetPtMin_ =cfg.getParameter<double> ("RecJetPtMin");
  GenJetPtBins_ =cfg.getParameter< std::vector<double> >("GenJetPtBins");
  std::vector<double> EtaBins =cfg.getParameter< std::vector<double> >("RecJetEtaBins");

  int nb=EtaBins.size();
  for(int i=nb;i>1;i--){RecJetEtaBins_.push_back(-1.0*EtaBins[i-1]);}
  for(int i=0;i<nb;i++){RecJetEtaBins_.push_back(EtaBins[i]);}

  histogramFile_= cfg.getParameter<std::string> ("HistogramFile"),

  hist_file_=new TFile(histogramFile_.c_str(),"RECREATE");

  NPtBins = GenJetPtBins_.size()-1;
  NEtaBins = RecJetEtaBins_.size()-1;

  bookHistograms();

}
void SimJetResponseAnalysis::endJob() {
  done();
}

void SimJetResponseAnalysis::analyze(edm::Event const& event, edm::EventSetup const& iSetup) {

  // To get information from the event setup, you must request the "Record"
  // which contains it and then extract the object you need
  //  edm::ESHandle<CaloGeometry> geometry;
  //  iSetup.get<IdealGeometryRecord>().get(geometry);

  // These declarations create handles to the types of records that you want
  // to retrieve from event "event".

  // We assume that GenJet are sorted in Pt

  edm::Handle<GenJetCollection> genjets;
  edm::Handle<CaloJetCollection> recjets;
  edm::Handle<GenMETCollection> genmet;
  edm::Handle<CaloMETCollection> recmet;

  event.getByLabel (genjets_,genjets);
  event.getByLabel (recjets_,recjets);
  event.getByLabel (genmet_,genmet);
  event.getByLabel (recmet_,recmet);

  analyze(*genjets,*recjets,*genmet,*recmet);
}


SimJetResponseAnalysis::SimJetResponseAnalysis() {
  hist_file_=0; // set to null
}

////////////////////////////////////////////////////////////////////////////////////////
void SimJetResponseAnalysis::fillHist1D(const TString& histName,const Double_t& value, const Double_t& wt) {

  std::map<TString, TH1*>::iterator hid=m_HistNames1D.find(histName);
  if (hid==m_HistNames1D.end()){
    std::cout << "%fillHist -- Could not find histogram with name: " << histName << std::endl;
  }
  else{
    hid->second->Fill(value,wt);
  }
}
/////////////////////////////////////////////////////////////////////////////////////////

void SimJetResponseAnalysis::fillHist2D(const TString& histName, const Double_t& x,const Double_t& y,const Double_t& wt) {

  std::map<TString, TH2*>::iterator hid=m_HistNames2D.find(histName);
  if (hid==m_HistNames2D.end()){
    // std::cout << "%fillHist -- Could not find histogram with name: " << histName << std::endl;
  }
  else {
    hid->second->Fill(x,y,wt);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////
void SimJetResponseAnalysis::bookHistograms() {

  bookGeneralHistograms();
  bookJetHistograms("Gen");
  bookJetHistograms("Calo");
  bookMetHists("Gen");
  bookMetHists("Calo");
  bookSimJetResponse();



}


void SimJetResponseAnalysis::analyze(const GenJetCollection& genjets,const CaloJetCollection& calojets,
					const GenMETCollection& genmets,const CaloMETCollection& recmets){
  fillJetHists(genjets,"Gen");
  fillJetHists(calojets,"Calo");
  fillMetHists(genmets,"Gen");
  fillMetHists(recmets,"Calo");

  SimulatedJetResponse(genjets,calojets);
}

////////////////////////////////////////////////////////////////////////////////////
void SimJetResponseAnalysis::bookGeneralHistograms() {

  TString hname;
  TString htitle;

  hname= "nevents";

  m_HistNames1D[hname] = new TH1F(hname,hname,20,0.0,20.0);
  m_HistNames1D[hname]->GetXaxis()->SetBinLabel(1,"Total");
  m_HistNames1D[hname]->GetXaxis()->SetBinLabel(2,"TwoJets");
  m_HistNames1D[hname]->GetXaxis()->SetBinLabel(3,"Central");
  m_HistNames1D[hname]->GetXaxis()->SetBinLabel(4,"AvePt");
  m_HistNames1D[hname]->GetXaxis()->SetBinLabel(5,"DeltaPhi");
  m_HistNames1D[hname]->GetXaxis()->SetBinLabel(6,"ThirdJet");

  m_HistNames1D[hname]->GetXaxis()->SetBinLabel(7,"CentralTrig0");
  m_HistNames1D[hname]->GetXaxis()->SetBinLabel(8,"CentralTrig1");


}

/////////////////////////////////////////////////////////////////////////////////////

void SimJetResponseAnalysis::bookJetHistograms(const TString& prefix) {

  TString hname;
  TString htitle;

  hname=prefix + "NumberOfJet";
  htitle=hname;
  m_HistNames1D[hname] = new TH1F(hname,htitle,100,-.0,100.0);

  for(int i=0; i<5; i++) {
    std::ostringstream oi; oi << i;
    hname=prefix + "PhiJet"+oi.str();
    htitle=hname;
    m_HistNames1D[hname] = new TH1F(hname,htitle,72,-M_PI,M_PI);

    hname=prefix + "EtJet"+oi.str();
    m_HistNames1D[hname]= new TH1F(hname,hname,500,0.0,5000.);
    hname=prefix + "PtJet"+oi.str();
    m_HistNames1D[hname]= new TH1F(hname,hname,500,0.0,5000.);

    hname=prefix + "RapidityJet"+oi.str();
    m_HistNames1D[hname] = new TH1F(hname,hname,100,-5.0,5.0);
  }
}

///////////////////////////////////////////////////////////////////////////////////

template <typename T> void SimJetResponseAnalysis::fillJetHists(const T& jets, const TString& prefix) {

  typedef typename T::const_iterator iter;
  TString hname;

  int NumOfJets=jets.size();

  hname=prefix + "NumberOfJet";
  fillHist1D(hname,NumOfJets);

  int ijet(0);

  for ( iter i=jets.begin(); i!=jets.end(); i++) {

    // First 5 jets only
    if(ijet>4) break;

    Double_t jetRapidity = i->y();
    Double_t jetPt = i->pt();
    Double_t jetEt = i->et();
    Double_t jetPhi = i->phi();

    std::ostringstream oi; oi << ijet;
   
    hname=prefix + "PhiJet"+oi.str();
    fillHist1D(hname,jetPhi);
    hname=prefix + "RapidityJet"+oi.str();
    fillHist1D(hname,jetRapidity);
    hname=prefix + "PtJet"+oi.str();
    fillHist1D(hname,jetPt);

    hname=prefix + "EtJet"+oi.str();
    fillHist1D(hname,jetEt);

    ijet++;
  }
}

void SimJetResponseAnalysis::bookMetHists(const TString& prefix) {

   TString hname;
   TString htitle;

   hname  =prefix+"NumberOfMetObjects";
   htitle =prefix+"Number of MET objects";
   m_HistNames1D[hname] = new TH1I(hname,htitle,10,0.,10.);

   hname=prefix+"etMiss";
   htitle=prefix+" Missing Et";
   m_HistNames1D[hname] = new TH1F(hname,htitle,100,0.0,100.);

   hname=prefix+"etMissX";
   htitle=prefix+" Missing Et-X";
   m_HistNames1D[hname] = new TH1F(hname,htitle,200,-100.0,100.);

   hname=prefix+"etMissPhi";
   htitle=prefix+" Phi of Missing Et";
   m_HistNames1D[hname] = new TH1F(hname,htitle,100,-M_PI,M_PI);

   hname=prefix+"sumEt";
   htitle=prefix+" Sum Et";
   m_HistNames1D[hname] = new TH1F(hname,htitle,1400,0.0,14000.);
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename T> void SimJetResponseAnalysis::fillMetHists(const T& mets, const TString& prefix) {

   Int_t NumMet=mets.size();
   TString hname;
   hname=prefix + "NumberOfMetObjects";
   fillHist1D(hname,NumMet);

   typedef typename T::const_iterator iter;
   for ( iter met=mets.begin(); met!=mets.end(); met++) {
     Double_t mEt=met->et();
     Double_t sumEt=met->sumEt();
     Double_t mEtPhi=met->phi();
     Double_t mEtX=met->px();

     fillHist1D(prefix+"etMiss",mEt);
     fillHist1D(prefix + "sumEt",sumEt);
     fillHist1D(prefix + "etMissX",mEtX);
     fillHist1D(prefix + "etMissPhi",mEtPhi);
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////
void SimJetResponseAnalysis::done() {

   if (hist_file_!=0) {
     GetSimJetResponse();
     hist_file_->Write(); 
     //   std::cout << "Output histograms written to: " << histogramFile_ << std::endl;
     delete hist_file_;
     hist_file_=0;      
   }
}

void SimJetResponseAnalysis::bookSimJetResponse() {

  TString hname;
  TString htitle;

  for(int ip=0;ip<NPtBins;ip++){
    std::ostringstream oip; oip << GenJetPtBins_[ip];

    int nBinPt(100);
    double PtMinBin=0.0;
    double PtMaxBin=7000.0;

    if(GenJetPtBins_[ip+1]<200){
      nBinPt=400;
      PtMaxBin=400.;
    }
    else if(GenJetPtBins_[ip+1]<1000){
      nBinPt=400;
      PtMaxBin=2000.;
    }
    else {
      nBinPt=500;
      PtMaxBin=10000.;
    }
    
    for(int ie=0;ie<NEtaBins;ie++){
      //      std::ostringstream oie; oie << RecJetEtaBins_[ie]; 

      std::ostringstream oie; oie << ie;
      hname="JetResponsePt"+oip.str()+"Eta"+oie.str();
      htitle="JetResponsePt"+oip.str()+"Eta"+oie.str();
      m_HistNames1D[hname] = new TH1F(hname,htitle,100,0.,2.0);

      hname="RminPt"+oip.str()+"Eta"+oie.str();
      htitle="Rmin"+oip.str()+"_"+oie.str();

      m_HistNames1D[hname] = new TH1F(hname,htitle,100,0.,2.0);

      hname="PtGenJetPt"+oip.str()+"Eta"+oie.str();
      htitle="PtGenJetPt"+oip.str()+"Eta"+oie.str();
      m_HistNames1D[hname] = new TH1F(hname,htitle,nBinPt,PtMinBin,PtMaxBin);

      hname="PtCaloJetPt"+oip.str()+"Eta"+oie.str();
      htitle="PtCaloJetPt"+oip.str()+"Eta"+oie.str();
      m_HistNames1D[hname] = new TH1F(hname,htitle,nBinPt,PtMinBin,PtMaxBin);

      hname="EtaGenJetPt"+oip.str()+"Eta"+oie.str();
      htitle="EtaGenJetPt"+oip.str()+"Eta"+oie.str();
      m_HistNames1D[hname] = new TH1F(hname,htitle,82,CaloTowerEtaBoundries);

      hname="EtaCaloJetPt"+oip.str()+"Eta"+oie.str();
      htitle="EtaCaloJetPt"+oip.str()+"Eta"+oie.str();
      m_HistNames1D[hname] = new TH1F(hname,htitle,82,CaloTowerEtaBoundries);

      //----------------------------


      //      std::ostringstream oie; oie << ie;
      hname="JetResponseEt"+oip.str()+"Eta"+oie.str();
      htitle="JetResponseEt"+oip.str()+"Eta"+oie.str();
      m_HistNames1D[hname] = new TH1F(hname,htitle,100,0.,2.0);

      hname="RminEt"+oip.str()+"Eta"+oie.str();
      htitle="Rmin"+oip.str()+"_"+oie.str();

      m_HistNames1D[hname] = new TH1F(hname,htitle,100,0.,2.0);

      hname="EtGenJetEt"+oip.str()+"Eta"+oie.str();
      htitle="EtGenJetEt"+oip.str()+"Eta"+oie.str();
      m_HistNames1D[hname] = new TH1F(hname,htitle,nBinPt,PtMinBin,PtMaxBin);

      hname="EtCaloJetEt"+oip.str()+"Eta"+oie.str();
      htitle="EtCaloJetEt"+oip.str()+"Eta"+oie.str();
      m_HistNames1D[hname] = new TH1F(hname,htitle,nBinPt,PtMinBin,PtMaxBin);

      hname="EtaGenJetEt"+oip.str()+"Eta"+oie.str();
      htitle="EtaGenJetEt"+oip.str()+"Eta"+oie.str();
      m_HistNames1D[hname] = new TH1F(hname,htitle,82,CaloTowerEtaBoundries);

      hname="EtaCaloJetEt"+oip.str()+"Eta"+oie.str();
      htitle="EtaCaloJetEt"+oip.str()+"Eta"+oie.str();
      m_HistNames1D[hname] = new TH1F(hname,htitle,82,CaloTowerEtaBoundries);
     
    }
    //  }

    //  for(int ip=0;ip<NPtBins;ip++){
    //    std::ostringstream oip; oip << GenJetPtBins_[ip];
    for(int it=0; it<NETA; it++){
      std::ostringstream oit; oit << it;
      hname="JetResponsePt"+oip.str()+"Tower"+oit.str();
      htitle="JetResponsePt"+oip.str()+"Tower"+oit.str();
      m_HistNames1D[hname] = new TH1F(hname,htitle,100,0.,2.0);

      hname="EtaCaloJetPt"+oip.str()+"Tower"+oit.str();
      htitle="EtaCaloJetPt"+oip.str()+"Tower"+oit.str();
      m_HistNames1D[hname] = new TH1F(hname,htitle,82,CaloTowerEtaBoundries);


      hname="EtaGenJetPt"+oip.str()+"Tower"+oit.str();
      htitle="EtaGenJetPt"+oip.str()+"Tower"+oit.str();
      m_HistNames1D[hname] = new TH1F(hname,htitle,82,CaloTowerEtaBoundries);

      hname="PtGenJetPt"+oip.str()+"Tower"+oit.str();
      htitle="PtGenJetPt"+oip.str()+"Tower"+oit.str();
      m_HistNames1D[hname] = new TH1F(hname,htitle,nBinPt,PtMinBin,PtMaxBin);

      hname="PtCaloJetPt"+oip.str()+"Tower"+oit.str();
      htitle="PtCaloJetPt"+oip.str()+"Tower"+oit.str();
      m_HistNames1D[hname] = new TH1F(hname,htitle,nBinPt,PtMinBin,PtMaxBin);

    }
  }

  for(int ip=0;ip<NPtBins;ip++){
    std::ostringstream oip; oip << GenJetPtBins_[ip];
    hname="ResponseVsEtaMean"+oip.str();
    htitle="ResponseVsEtaMean"+oip.str();
    m_HistNames1D[hname] = new TH1F(hname,htitle,82,CaloTowerEtaBoundries);

    hname="ResponseVsEtaMPV"+oip.str();
    htitle="ResponseVsEtaMPV"+oip.str();
    m_HistNames1D[hname] = new TH1F(hname,htitle,82,CaloTowerEtaBoundries);
  }
}
int  SimJetResponseAnalysis::GetPtBin(double GenJetPt){
  for(int ip=0;ip<NPtBins;ip++){
    if(GenJetPtBins_[ip] <GenJetPt && GenJetPt < GenJetPtBins_[ip+1] ){
      return ip;
    }
  }
  return 0;
}
int  SimJetResponseAnalysis::TowerNumber(double eta){
  for(int i=0;i<NETA;i++){
    if(CaloTowerEtaBoundries[i]<eta  && eta<=CaloTowerEtaBoundries[i+1]){
      return i;
    }
  }
  return 0;
}
int  SimJetResponseAnalysis::GetEtaBin(double eta){
   for(int ie=0;ie<NEtaBins;ie++){
    if(eta>= RecJetEtaBins_[ie] && eta< RecJetEtaBins_[ie+1]){
      return ie;
    }
  }
  return 0;
}

void SimJetResponseAnalysis::SimulatedJetResponse(const GenJetCollection& genjets,const CaloJetCollection& calojets){

  if(genjets.size()==0) return;
  if(calojets.size()==0) return;

  const double GenJetEtaMax=5.5;

  double GenJetPtMin=GenJetPtBins_[0];

  TString hname;

  int njet(0);

  for(GenJetIter i=genjets.begin();i!=genjets.end(); i++) {
    njet++;
    if(njet>NJetMax_) return;                     // only two leading jets 
    Double_t GenJetPt = i->pt();
    Double_t GenJetEt = i->et();
    Double_t GenJetEta = i->eta();
    if(GenJetPt>GenJetPtMin) {
      if(fabs(GenJetEta)<GenJetEtaMax){
	float rmin(99);
	CalJetIter caljet;
	for(CalJetIter j=calojets.begin();j!=calojets.end();j++){
	  float rr=radius(i,j);
	  if(rr<rmin){rmin=rr;caljet=j;}
	}
    
	double CaloJetPt=caljet->pt();
	double CaloJetEt=caljet->et();
	double CaloJetEta=caljet->eta();
	double ResponsePt=CaloJetPt/GenJetPt;
	double ResponseEt=CaloJetEt/GenJetEt;

        if(CaloJetPt<RecJetPtMin_) continue;

	int ipt = GetPtBin(GenJetPt);
	int iet = GetPtBin(GenJetEt);
	int ie = GetEtaBin(CaloJetEta);
	int it = TowerNumber(CaloJetEta);
	std::ostringstream oipt; oipt << GenJetPtBins_[ipt];
	std::ostringstream oiet; oiet << GenJetPtBins_[iet];
	//	std::ostringstream oie; oie << RecJetEtaBins_[ie];
	std::ostringstream oie; oie <<ie;
	std::ostringstream oit; oit << it;

	hname="RminPt"+oipt.str()+"Eta"+oie.str();
	fillHist1D(hname,rmin);
	if(rmin<MatchRadius_){
	    //	    cout << " radius " << rmin << " GenPt " << GenJetPt << " CalPt " << caljet->pt() <<" rr  " << calpt/GenJetPt <<endl;
	  hname="JetResponsePt"+oipt.str()+"Eta"+oie.str();
	  fillHist1D(hname,ResponsePt);

	  hname="EtaGenJetPt"+oipt.str()+"Eta"+oie.str();
	  fillHist1D(hname,GenJetEta);

	  hname="PtGenJetPt"+oipt.str()+"Eta"+oie.str();
	  fillHist1D(hname,GenJetPt);

	  hname="PtCaloJetPt"+oipt.str()+"Eta"+oie.str();
	  fillHist1D(hname,CaloJetPt);

	  hname="EtaCaloJetPt"+oipt.str()+"Eta"+oie.str();
	  fillHist1D(hname,CaloJetEta);

	  // Et plots==================================
 
	  hname="JetResponseEt"+oiet.str()+"Eta"+oie.str();
	  fillHist1D(hname,ResponseEt);

	  hname="EtaGenJetEt"+oiet.str()+"Eta"+oie.str();
	  fillHist1D(hname,GenJetEta);

	  hname="EtGenJetEt"+oiet.str()+"Eta"+oie.str();
	  fillHist1D(hname,GenJetEt);

	  hname="EtCaloJetEt"+oiet.str()+"Eta"+oie.str();
	  fillHist1D(hname,CaloJetEt);

	  hname="EtaCaloJetEt"+oiet.str()+"Eta"+oie.str();
	  fillHist1D(hname,CaloJetEta);

	  // Tower plots=====================================

	  hname="JetResponsePt"+oipt.str()+"Tower"+oit.str();
	  fillHist1D(hname,ResponsePt);

	  hname="EtaCaloJetPt"+oipt.str()+"Tower"+oit.str();
	  fillHist1D(hname,CaloJetEta);

	  hname="EtaGenJetPt"+oipt.str()+"Tower"+oit.str();
	  fillHist1D(hname,GenJetEta);

	  hname="PtGenJetPt"+oipt.str()+"Tower"+oit.str();
	  fillHist1D(hname,GenJetPt);

	  hname="PtCaloJetPt"+oipt.str()+"Tower"+oit.str();
	  fillHist1D(hname,CaloJetPt);
	}
      }
    }
  }
}
 
void GetMPV(TH1* h,double& mean,double& width,double& error) {
 
  mean=0.0;
  width=0.0;
  error=0.0;
  Double_t nevents=h->GetEntries();  
  if(nevents<1) return;

  h->SetNormFactor(1.0);

  //  Double_t median[1];
  //  Double_t half[1]={0.5};
  //  h->GetQuantiles(1,median,half);
  //  Double_t gausmean=median[0];

  Double_t gausmean=h->GetMean();
  Double_t gauswidth=h->GetRMS();
  Double_t norm=1.0;
 
  TF1 *g = new TF1("g","gaus(0)",gausmean-2.0*gauswidth,gausmean+2.0*gauswidth);
  g->SetParLimits(0,0.0,2.0);
  g->SetParameter(0,norm);
  g->SetParameter(1,gausmean);
  g->SetParameter(2,gauswidth);
  h->Fit(g,"RQN");
 
  norm=g->GetParameter(0);
  gausmean=g->GetParameter(1);
  gauswidth=g->GetParameter(2);
 
  g->SetRange(gausmean-1.0*gauswidth,gausmean+1.0*gauswidth);
  g->SetParameter(0,norm);
  g->SetParameter(1,gausmean);
  g->SetParameter(2,gauswidth);
  h->Fit(g,"RQN");
 
  norm=g->GetParameter(0);
  gausmean=g->GetParameter(1);
  gauswidth=g->GetParameter(2);
 
  g->SetRange(gausmean-1.0*gauswidth,gausmean+1.0*gauswidth);
  g->SetParameter(0,norm);
  g->SetParameter(1,gausmean);
  g->SetParameter(2,gauswidth);
  h->Fit(g,"RQ");

  mean=g->GetParameter(1);
  width=g->GetParameter(2);
  error=width/sqrt(nevents);

}

void  SimJetResponseAnalysis::GetSimJetResponse(){

  TString hname;

  std::map<TString, TH1*>::iterator h1D;

  const int MinimumEvents=25;

  for(int ip=0;ip<NPtBins;ip++){
    std::ostringstream oip; oip << GenJetPtBins_[ip];


    std::vector<double> x(NETA);
    std::vector<double> y(NETA);
    std::vector<double> n(NETA);
    std::vector<double> m(NETA);
    std::vector<double> e(NETA);

    std::vector<double> fittedMean(NETA);
    std::vector<double> fittedWidth(NETA);
    std::vector<double> fittedError(NETA);

    std::vector<double> neve(NETA);
    std::vector<double> ex(NETA);
    std::vector<double> ey(NETA);

    int np=0;
    for(int it=0; it<NETA; it++){
      x[np] = (CaloTowerEtaBoundries[it]+CaloTowerEtaBoundries[it+1])/2.0;
      std::ostringstream oit; oit << it;
      hname="JetResponsePt"+oip.str()+"Tower"+oit.str();
      h1D=m_HistNames1D.find(hname);

      if(h1D!=m_HistNames1D.end()){
	int nevents = int(h1D->second->GetEntries());
	n[it] = nevents;
	//      std::cout << " Number of events " <<" itower "<< it << " nevents " << nevents << std::endl;
	if(nevents>MinimumEvents){     
	  m[it] = h1D->second->GetMean();
	  e[it] = h1D->second->GetRMS()/sqrt(n[it]);
	  GetMPV(h1D->second,fittedMean[it],fittedWidth[it],fittedError[it]);
	  // cout << "itower " << it <<" Nevents " << nevents<< " Mean " << m[it]<< " Fitted Mean " << fittedMean[it] <<endl;
	}
        else{
	  m[it] =0.0;
	  e[it] = 0.0;
	  fittedMean[it] =0.0;
	  fittedWidth[it] =0.0;
	  fittedError[it] =0.0;
	}
      }
      else {
	//        cout << hname << " does not exist " <<endl;
      }

    }

    hname="ResponseVsEtaMean"+oip.str();
    h1D=m_HistNames1D.find(hname);

    for(int i=0;i<NETA;i++){
      h1D->second->SetBinContent(i,m[i]);
      h1D->second->SetBinError(i,e[i]);
    }

    hname="ResponseVsEtaMPV"+oip.str();
    h1D=m_HistNames1D.find(hname);

    for(int i=0;i<NETA;i++){
      h1D->second->SetBinContent(i,fittedMean[i]);
      h1D->second->SetBinError(i,fittedError[i]);
    }
  }
}


#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SimJetResponseAnalysis);
