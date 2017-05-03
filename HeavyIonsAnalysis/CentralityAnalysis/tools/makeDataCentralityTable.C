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
#include "DataFormats/FWLite/interface/ChainEvent.h"

#include "DataFormats/HeavyIonEvent/interface/CentralityBins.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/TriggerResults.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"

#endif

void fitSlices(TH2*, TF1*);

static bool useEfficiencyHistogram = false;
static bool onlySaveTable = false;
static const int nbinsMax = 40;
static bool doTrigger = false;
static int trigNum = 1; // 6 for Hydjet 2.8 TeV sample

using namespace std;
bool descend(float i,float j) { return (i>j); }

void makeDataCentralityTable(int nbins = 40, const string label = "hf", const char * datatag = "HFhitBins",const char * mctag = "HFhitBinsMC", double EFF = 1){

   // This macro assumes all inefficiency is in the most peripheral bin.
   double MXS = 1. - EFF;
   int nFiles = 1;
   vector<string> infiles;
   //   TFile* infile = new TFile("../data/Centrality0705_merged_runs1to10.root"); 

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

  // Retrieving data
  // Creating output table
  TFile * centFile = new TFile("../data/CentralityTables.root","update");
  TH1* hEff;
  if(useEfficiencyHistogram) hEff = (TH1*)centFile->Get(Form("%s/hEff",mctag));
  TDirectory* dir = centFile->mkdir(datatag);
  dir->cd();

  TH1D::SetDefaultSumw2();
  int runMC = 1;

  CentralityBins::RunMap HFhitBinMap = getCentralityFromFile(centFile,mctag,0,20);
  nbins = HFhitBinMap[runMC]->getNbins();

  CentralityBins* bins = new CentralityBins("noname","Test tag", nbins);
  bins->table_.reserve(nbins);

  // Setting up variables & branches
  double binboundaries[nbinsMax+1];
  vector<float> values;

  // Determining bins of cross section
  // loop over events
  unsigned int events=0;
  double xsec = 0;

  for(event.toBegin(); !event.atEnd(); ++event, ++events){
     edm::EventBase const & ev = event;
    if( events % 100 == 0 ) cout<<"Processing event : "<<events<<endl;
    edm::Handle<reco::Centrality> cent;
    ev.getByLabel(edm::InputTag("hiCentrality"),cent);
    edm::Handle<edm::TriggerResults> trig;
    if(doTrigger){
       ev.getByLabel(edm::InputTag("TriggerResults","","HLT"),trig);
       bool t = trig->at(trigNum).accept();
       if(!t) continue;
    }

    double hf = cent->EtHFhitSum();
    double hftp = cent->EtHFtowerSumPlus();
    double hftm = cent->EtHFtowerSumMinus();
    double eb = cent->EtEBSum();
    double eep = cent->EtEESumPlus();
    double eem = cent->EtEESumMinus();
    double parameter = 0;
    if(label.compare("hf") == 0) parameter = hf;
    if(label.compare("hft") == 0) parameter = hftp + hftm;
    if(label.compare("eb") == 0) parameter = eb;
    if(label.compare("ee") == 0) parameter = eep+eem;
    values.push_back(parameter);

    // Calculate corrected cross section
    if(useEfficiencyHistogram){
       xsec += 1. / hEff->GetBinContent(hEff->FindBin(parameter));
    }else{
       xsec += 1.;
    }

    int run = event.id().run();
    if(runnums.size() == 0 || runnums[runnums.size()-1] != run) runnums.push_back(run);
  }
  
  if(label.compare("b") == 0) sort(values.begin(),values.end());
  else sort(values.begin(),values.end(),descend);

  double max = values[events-1];
  binboundaries[nbins] = max;

  cout<<"-------------------------------------"<<endl;
  cout<<label.data()<<" based cuts are : "<<endl;
  cout<<"(";

  double integral = 0;
  int currentbin = 0;
  for(int iv = 0; iv < events && currentbin < nbins; ++iv){
     double val = values[iv];  
     if(useEfficiencyHistogram){
	integral += 1 / hEff->GetBinContent(hEff->FindBin(val));
     }else{
	integral += 1;
     }
     if(integral > (int)((currentbin+1)*(xsec/nbins))){
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
     bins->table_[i].n_part_mean = HFhitBinMap[runMC]->NpartMeanOfBin(i);
     bins->table_[i].n_part_var = HFhitBinMap[runMC]->NpartSigmaOfBin(i); 
     bins->table_[i].n_coll_mean = HFhitBinMap[runMC]->NcollMeanOfBin(i);
     bins->table_[i].n_coll_var = HFhitBinMap[runMC]->NcollSigmaOfBin(i);
     bins->table_[i].b_mean = HFhitBinMap[runMC]->bMeanOfBin(i);
     bins->table_[i].b_var = HFhitBinMap[runMC]->bSigmaOfBin(i);
     bins->table_[i].n_hard_mean = HFhitBinMap[runMC]->NhardMeanOfBin(i);
     bins->table_[i].n_hard_var = HFhitBinMap[runMC]->NhardSigmaOfBin(i);
     bins->table_[i].bin_edge = binboundaries[i];

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
  centFile->cd();
  dir->cd();
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





