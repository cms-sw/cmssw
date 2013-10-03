#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <map>
#include <TF1.h>
#include <TH2D.h>
#include <TH1D.h>
#include <TNtuple.h>
#include <TFile.h>
#include <TSystem.h>

#if !defined(__CINT__) && !defined(__MAKECINT__)


#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/HeavyIonEvent/interface/CentralityBins.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/FWLite/interface/ChainEvent.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"

#endif

void fitSlices(TH2*, TF1*);

static bool onlySaveTable = false;
static const int nbinsMax = 40;

using namespace std;
bool descend(float i,float j) { return (i<j); }

void makeCentralityTable(int nbins = 40, const string label = "hf", const char * tag = "HFhitBins", double MXS = 0.){

   // This macro assumes all inefficiency is in the most peripheral bin.
   double EFF = 1. - MXS;

   // Retrieving data
  int nFiles = 1;
  vector<string> infiles;
  //  TFile* infile = new TFile("/net/hisrv0001/home/yetkin/pstore02/ana/Hydjet_MinBias_d20100222/Hydjet_MinBias_4TeV_runs1to300.root");
  //  fwlite::Event event(infile);
  infiles.push_back("~/hibat0007/aod/JulyExercise/MinBias0707/MinBias0707_runs1to10.root");
  //  infiles.push_back("~/hibat0007/aod/JulyExercise/MinBias0707/MinBias0707_runs11to20.root");
  infiles.push_back("~/hibat0007/aod/JulyExercise/MinBias0707/MinBias0707_runs21to30.root");
  infiles.push_back("~/hibat0007/aod/JulyExercise/MinBias0707/MinBias0707_runs31to40.root");
  infiles.push_back("~/hibat0007/aod/JulyExercise/MinBias0707/MinBias0707_runs41to50.root");
  infiles.push_back("~/hibat0007/aod/JulyExercise/MinBias0707/MinBias0707_runs51to60.root");
  //  infiles.push_back("~/hibat0007/aod/JulyExercise/MinBias0707/MinBias0707_runs61to70.root");
  //  infiles.push_back("~/hibat0007/aod/JulyExercise/MinBias0707/MinBias0707_runs71to80.root");
  //  infiles.push_back("~/hibat0007/aod/JulyExercise/MinBias0707/MinBias0707_runs81to90.root")
  //  infiles.push_back("~/hibat0007/aod/JulyExercise/MinBias0707/MinBias0707_runs91to100.root");

  fwlite::ChainEvent event(infiles);

  vector<int> runnums;

  // Creating output table
  TFile* outFile = new TFile("tables.root","update");
   TDirectory* dir = outFile->mkdir(tag);
   dir->cd();

  TH1D::SetDefaultSumw2();
  CentralityBins* bins = new CentralityBins("noname","Test tag", nbins);
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
    
    int run = event.id().run();
    if(runnums.size() == 0 || runnums[runnums.size()-1] != run) runnums.push_back(run);
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
     double xsec = events*(1 + MXS);
     // Below should be replaced with an integral
     // when inefficiency is better parametrized 
     // than a step function.

     int entry = (int)(i*(xsec/nbins)) - offset;
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


  TF1* fGaus = new TF1("fb","gaus(0)",0,2); 
  fGaus->SetParameter(0,1);
  fGaus->SetParameter(1,0.04);
  fGaus->SetParameter(2,0.02); 
  
  fitSlices(hNpart,fGaus);
  fitSlices(hNcoll,fGaus);
  fitSlices(hNhard,fGaus);
  fitSlices(hb,fGaus);

 /*
  hNpart->FitSlicesY();
  hNcoll->FitSlicesY();
  hNhard->FitSlicesY();
  hb->FitSlicesY();
 */

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

     TH1D* hh = (TH1D*)gDirectory->Get("hNpart_0");
     hh->Delete();
     hh = (TH1D*)gDirectory->Get("hNcoll_0");
     hh->Delete();
     hh = (TH1D*)gDirectory->Get("hNhard_0");
     hh->Delete();
     hh = (TH1D*)gDirectory->Get("hb_0");
     hh->Delete();

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
  
  for(int i = 0; i < runnums.size(); ++i){
     CentralityBins* binsForRun = (CentralityBins*) bins->Clone();
     binsForRun->SetName(Form("run%d",runnums[i]));
     binsForRun->Write();
  }
  
  bins->Delete();
  outFile->Write();
  
}

void fitSlices(TH2* hCorr, TF1* func){

   int nBins = hCorr->GetNbinsX();

   TH1D* hMean = new TH1D(Form("%s_1",hCorr->GetName()),"",nBins,hCorr->GetXaxis()->GetXmin(),hCorr->GetXaxis()->GetXmax());
   TH1D* hSigma = new TH1D(Form("%s_2",hCorr->GetName()),"",nBins,hCorr->GetXaxis()->GetXmin(),hCorr->GetXaxis()->GetXmax());

   for(int i = 1; i < nBins; ++i){
      TH1D* h = hCorr->ProjectionY(Form("%s_bin%d",hCorr->GetName(),i),i-1,i);
      h->Fit(func);
      hMean->SetBinContent(i,func->GetParameter(1));
      hMean->SetBinError(i,func->GetParError(1));
      hSigma->SetBinContent(i,func->GetParameter(2));
      hSigma->SetBinError(i,func->GetParError(2));
      if(onlySaveTable){
	 h->Delete();
      }
   }
}





