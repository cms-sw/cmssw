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
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/Common/interface/TriggerResults.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"

#endif

void fitSlices(TH2*, TF1*);

static bool onlySaveTable = false;
static const int nbinsMax = 40;
static bool doTrigger = true;
static int trigNum = 1; // 6 for Hydjet 2.8 TeV sample

using namespace std;
bool descend(float i,float j) { return (i<j); }

void makeDataCentralityTable(int nbins = 40, const string label = "hf", const char * datatag = "HFhitBins",const char * mctag = "HFhitBinsMC", double EFF = 0.88){

   // This macro assumes all inefficiency is in the most peripheral bin.
   double MXS = 1. - EFF;
  int nFiles = 1;
  vector<string> fileNames;
  TFile* infile = new TFile("/net/hisrv0001/home/yetkin/pstore02/ana/Hydjet_MinBias_4TeV_d20100305/Hydjet_MinBias_4TeV_runs1to500.root");
  fwlite::Event event(infile);
  vector<int> runnums;

  // Retrieving data
  // Creating output table
  TFile * centFile = new TFile("../data/CentralityTables.root","update");
  TH1* hEff = (TH1*)centFile->Get(Form("%s/hEff",mctag));
  TDirectory* dir = centFile->mkdir(datatag);
  dir->cd();

  TH1D::SetDefaultSumw2();
  int runMC = 1;
  CentralityBins::RunMap HFhitBinMap = getCentralityFromFile(centFile,mctag,runMC - 1,runMC + 1);
  nbins = HFhitBinMap[runMC]->getNbins();
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
    edm::Handle<edm::TriggerResults> trig;
    ev.getByLabel(edm::InputTag("TriggerResults","","HLT"),trig);

    bool t = trig->at(trigNum).accept();
    if(doTrigger && !t) continue;
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

  double integral = 0;
  int currentbin = 0;
  for(int iv = 0; iv < events && currentbin < nbins; ++iv){
     double val = values[iv];  
     integral += val / hEff->GetBinContent(hEff->FindBin(val));
     if(integral > (int)(currentbin*(events/nbins))){
	binboundaries[currentbin] = val;
	cout<<" "<<val;
	if(currentbin < nbins - 1) cout<<",";
	else cout<<")"<<endl;

	currentbin++;
     }
  }
  cout<<"-------------------------------------"<<endl;

  // Retrieving Glauber results in various bins
  // from input MC centrality table
  cout<<"-------------------------------------"<<endl;
  cout<<"# Bin NpartMean NpartSigma NcollMean NcollSigma bMean bSigma BinEdge"<<endl;

  // Enter values in table
  for(int i = 0; i < nbins; ++i){
     bins->table_[nbins-i-1].n_part_mean = HFhitBinMap[runMC]->NpartMeanOfBin(i);
     bins->table_[nbins-i-1].n_part_var = HFhitBinMap[runMC]->NpartSigmaOfBin(i); 
     bins->table_[nbins-i-1].n_coll_mean = HFhitBinMap[runMC]->NcollMeanOfBin(i);
     bins->table_[nbins-i-1].n_coll_var = HFhitBinMap[runMC]->NcollSigmaOfBin(i);
     bins->table_[nbins-i-1].b_mean = HFhitBinMap[runMC]->bMeanOfBin(i);
     bins->table_[nbins-i-1].b_var = HFhitBinMap[runMC]->bSigmaOfBin(i);
     bins->table_[nbins-i-1].n_hard_mean = HFhitBinMap[runMC]->NhardMeanOfBin(i);
     bins->table_[nbins-i-1].n_hard_var = HFhitBinMap[runMC]->NhardSigmaOfBin(i);
     bins->table_[nbins-i-1].bin_edge = binboundaries[i];

     cout<<i<<" "
	 <<HFhitBinMap[runMC]->NpartMeanOfBin(i)<<" "
	 <<HFhitBinMap[runMC]->NpartSigmaOfBin(i)<<" "
	 <<HFhitBinMap[runMC]->NcollMeanOfBin(i)<<" "
	 <<HFhitBinMap[runMC]->NcollSigmaOfBin(i)<<" "
	 <<HFhitBinMap[runMC]->bMeanOfBin(i)<<" "
	 <<HFhitBinMap[runMC]->bSigmaOfBin(i)<<" "
	 <<binboundaries[i]<<" "
	 <<endl;
  }
  cout<<"-------------------------------------"<<endl;

  // Save the table in output file
  for(int i = 0; i < runnums.size(); ++i){
     CentralityBins* binsForRun = (CentralityBins*) bins->Clone();
     binsForRun->SetName(Form("run%d",runnums[i]));
     binsForRun->Write();
  }
  
  bins->Delete();
  centFile->Write();
  
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





