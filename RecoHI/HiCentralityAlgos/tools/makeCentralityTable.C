#include <memory>
#include <string>
#include <vector>
#include <iostream>

#include <TH2D.h>
#include <TH1D.h>
#include <TNtuple.h>
#include <TFile.h>
#include <TSystem.h>

#if !defined(__CINT__) && !defined(__MAKECINT__)


#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/HeavyIonEvent/interface/CentralityBins.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"

#endif

static const int nbinsMax = 40;
using namespace std;
bool descend(float i,float j) { return (i<j); }

void makeCentralityTable(int nbins = 40, const string label = "hf", const char * tag = "HFhitBins", double MXS = 0.){

   // This macro assumes all inefficiency is in the most peripheral bin.
   double EFF = 1. - MXS;
   bool onlySaveTable = true;

   // Retrieving data
  int nFiles = 1;
  vector<string> fileNames;
  TFile* infile = new TFile("/net/hisrv0001/home/yetkin/pstore02/ana/Hydjet_MinBias_d20100222/Hydjet_MinBias_4TeV_runs1to300.root");
  fwlite::Event event(infile);

  // Creating output table
  TFile* outFile = new TFile("tables.root","update");
  TH1D::SetDefaultSumw2();
  CentralityBins* bins = new CentralityBins(tag,"Test tag", nbins);
  bins->table_.reserve(nbins);

  // Setting up variables & branches
  double binboundaries[nbinsMax+1];
  vector<float> values;

  // Determining bins of cross section
  // loop over events
  unsigned int events=0;
  for(event.toBegin(); !event.atEnd(); ++event, ++events){
     edm::EventBase const & ev = event;
    if( events % 100 == 0 ) cout<<"Processing event : "<<events<<endl;
    edm::Handle<edm::GenHIEvent> mc;
    ev.getByLabel(edm::InputTag("heavyIon"),mc);
    edm::Handle<reco::Centrality> cent;
    ev.getByLabel(edm::InputTag("hiCentrality"),cent);

    double b = mc->b();
    double npart = mc->Npart();
    double ncoll = mc->Ncoll();
    double nhard = mc->Nhard();

    double hf = cent->EtHFhitSum();
    double hftp = cent->EtHFtowerSumPlus();
    double hftm = cent->EtHFtowerSumMinus();
    double eb = cent->EtEBSum();
    double eep = cent->EtEESumPlus();
    double eem = cent->EtEESumMinus();

    double parameter = 0;
    if(label.compare("npart") == 0) parameter = npart;
    if(label.compare("ncoll") == 0) parameter = ncoll;
    if(label.compare("nhard") == 0) parameter = nhard;
    if(label.compare("b") == 0) parameter = b;
    if(label.compare("hf") == 0) parameter = hf;
    if(label.compare("hft") == 0) parameter = hftp + hftm;
    if(label.compare("eb") == 0) parameter = eb;
    if(label.compare("ee") == 0) parameter = eep+eem;

    values.push_back(parameter);
  }

  if(label.compare("b") == 0) sort(values.begin(),values.end(),descend);
  else sort(values.begin(),values.end());

  double max = values[events-1];
  binboundaries[nbins] = max;

  cout<<"-------------------------------------"<<endl;
  cout<<label.data()<<" based cuts are : "<<endl;
  cout<<"(";

  int bin = 0;
  for(int i = 0; i< nbins; ++i){
     // Find the boundary 
     int offset = (int)(MXS*events);
     double xsec = events*EFF;
     // Below should be replaced with an integral
     // when inefficiency is better parametrized 
     // than a step function.

     int entry = (int)(i*(xsec/nbins)) + offset;
     binboundaries[i] = values[entry];

     cout<<" "<<binboundaries[i];
     if(i < nbins - 1) cout<<",";
     else cout<<")"<<endl;
  }
  cout<<"-------------------------------------"<<endl;

  // Determining Glauber results in various bins
  TH2D* hNpart = new TH2D("hNpart","",nbins,binboundaries,500,0,500);
  TH2D* hNcoll = new TH2D("hNcoll","",nbins,binboundaries,2000,0,2000);
  TH2D* hNhard = new TH2D("hNhard","",nbins,binboundaries,250,0,250);
  TH2D* hb = new TH2D("hb","",nbins,binboundaries,300,0,30);

  for(event.toBegin(); !event.atEnd(); ++event){
     edm::EventBase const & ev = event;
     edm::Handle<edm::GenHIEvent> mc;
     ev.getByLabel(edm::InputTag("heavyIon"),mc);
     edm::Handle<reco::Centrality> cent;
     ev.getByLabel(edm::InputTag("hiCentrality"),cent);

     double b = mc->b();
     double npart = mc->Npart();
     double ncoll = mc->Ncoll();
     double nhard = mc->Nhard();

     double hf = cent->EtHFhitSum();
     double hftp = cent->EtHFtowerSumPlus();
     double hftm = cent->EtHFtowerSumMinus();
     double eb = cent->EtEBSum();
     double eep = cent->EtEESumPlus();
     double eem = cent->EtEESumMinus();

     double parameter = 0;

     if(label.compare("npart") == 0) parameter = npart;
     if(label.compare("ncoll") == 0) parameter = ncoll;
     if(label.compare("nhard") == 0) parameter = nhard;
     if(label.compare("b") == 0) parameter = b;
     if(label.compare("hf") == 0) parameter = hf;
     if(label.compare("hft") == 0) parameter = hftp + hftm;
     if(label.compare("eb") == 0) parameter = eb;
     if(label.compare("ee") == 0) parameter = eep+eem;
    
     hNpart->Fill(parameter,npart);
     hNcoll->Fill(parameter,ncoll);
     hNhard->Fill(parameter,nhard);
     hb->Fill(parameter,b);
  }

  // Fitting Glauber distributions in bins to get mean and sigma values
  hNpart->FitSlicesY();
  hNcoll->FitSlicesY();
  hNhard->FitSlicesY();
  hb->FitSlicesY();

  TH1D* hNpartMean = (TH1D*)gDirectory->Get("hNpart_1");
  TH1D* hNpartSigma = (TH1D*)gDirectory->Get("hNpart_2");
  TH1D* hNcollMean = (TH1D*)gDirectory->Get("hNcoll_1");
  TH1D* hNcollSigma = (TH1D*)gDirectory->Get("hNcoll_2");
  TH1D* hNhardMean = (TH1D*)gDirectory->Get("hNhard_1");
  TH1D* hNhardSigma = (TH1D*)gDirectory->Get("hNhard_2");
  TH1D* hbMean = (TH1D*)gDirectory->Get("hb_1");
  TH1D* hbSigma = (TH1D*)gDirectory->Get("hb_2");

  cout<<"-------------------------------------"<<endl;
  cout<<"# Bin NpartMean NpartSigma NcollMean NcollSigma bMean bSigma BinEdge"<<endl;


  // Enter values in table
  for(int i = 0; i < nbins; ++i){
     bins->table_[nbins-i-1].n_part_mean = hNpartMean->GetBinContent(i);
     bins->table_[nbins-i-1].n_part_var = hNpartSigma->GetBinContent(i);
     bins->table_[nbins-i-1].n_coll_mean = hNcollMean->GetBinContent(i);
     bins->table_[nbins-i-1].n_coll_var = hNcollSigma->GetBinContent(i);
     bins->table_[nbins-i-1].b_mean = hbMean->GetBinContent(i);
     bins->table_[nbins-i-1].b_var = hbSigma->GetBinContent(i);
     bins->table_[nbins-i-1].n_hard_mean = hNhardMean->GetBinContent(i);
     bins->table_[nbins-i-1].n_hard_var = hNhardSigma->GetBinContent(i);
     bins->table_[nbins-i-1].bin_edge = binboundaries[i];

     cout<<i<<" "
	 <<hNpartMean->GetBinContent(i)<<" "
	 <<hNpartSigma->GetBinContent(i)<<" "
	 <<hNcollMean->GetBinContent(i)<<" "
	 <<hNcollSigma->GetBinContent(i)<<" "
	 <<hbMean->GetBinContent(i)<<" "
	 <<hbSigma->GetBinContent(i)<<" "
	 <<binboundaries[i]<<" "
	 <<endl;
  }
  cout<<"-------------------------------------"<<endl;

  // Save the table in output file

  if(onlySaveTable){
     hNpart->Delete();
     hNpartMean->Delete();
     hNpartSigma->Delete();
     hNcoll->Delete();
     hNcollMean->Delete();
     hNcollSigma->Delete();
     hNhard->Delete();
     hNhardMean->Delete();
     hNhardSigma->Delete();
     hb->Delete();
     hbMean->Delete();
     hbSigma->Delete();
  }
  
  bins->Write();
  outFile->Write();
  
}
