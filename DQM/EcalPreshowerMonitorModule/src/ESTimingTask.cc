#include <memory>
#include <fstream>
#include <iostream>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "CondFormats/DataRecord/interface/ESGainRcd.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/EcalPreshowerMonitorModule/interface/ESTimingTask.h"

#include "TMath.h"
#include "TGraph.h"

using namespace cms;
using namespace edm;
using namespace std;

// fit function
double fitf(double *x, double *par) {

  double wc = par[2];
  double n  = par[3]; // n-1 (in fact)
  double v1 = pow(wc/n*(x[0]-par[1]), n);
  double v2 = TMath::Exp(n-wc*(x[0]-par[1]));
  double v  = par[0]*v1*v2;

  if (x[0] < par[1]) v = 0;

  return v;
}

ESTimingTask::ESTimingTask(const edm::ParameterSet& ps) {

  digilabel_   = consumes<ESDigiCollection>(ps.getParameter<InputTag>("DigiLabel"));
  prefixME_    = ps.getUntrackedParameter<string>("prefixME", "EcalPreshower"); 
  
  eCount_ = 0;

  fit_ = new TF1("fitShape", fitf, -200, 200, 4);
  fit_->SetParameters(50, 10, 0, 0);
  
  //Histogram init  
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) 
      hTiming_[i][j] = 0;

  htESP_ = new TH1F("htESP", "Timing ES+", 81, -20.5, 20.5);
  htESM_ = new TH1F("htESM", "Timing ES-", 81, -20.5, 20.5);
}

void
ESTimingTask::bookHistograms(DQMStore::IBooker& iBooker, Run const&, EventSetup const&)
{
  iBooker.setCurrentFolder(prefixME_ + "/ESTimingTask");
  
  //Booking Histograms
  //Notice: Change ESRenderPlugin under DQM/RenderPlugins/src if you change this histogram name.
  char histo[200];
  for (int i=0 ; i<2; ++i) 
    for (int j=0 ; j<2; ++j) {
      int iz = (i==0)? 1:-1;
      sprintf(histo, "ES Timing Z %d P %d", iz, j+1);
      hTiming_[i][j] = iBooker.book1D(histo, histo, 81, -20.5, 20.5);
      hTiming_[i][j]->setAxisTitle("ES Timing (ns)", 1);
    }

  sprintf(histo, "ES 2D Timing");
  h2DTiming_ = iBooker.book2D(histo, histo, 81, -20.5, 20.5, 81, -20.5, 20.5);
  h2DTiming_->setAxisTitle("ES- Timing (ns)", 1);
  h2DTiming_->setAxisTitle("ES+ Timing (ns)", 2);
}

ESTimingTask::~ESTimingTask() {
  delete fit_;
  delete htESP_;
  delete htESM_;
}

void ESTimingTask::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  
  set(iSetup);

  runNum_ = e.id().run();
  eCount_++;
  
  htESP_->Reset();
  htESM_->Reset();
  
  //Digis
  int zside, plane, ix, iy, is;
  double adc[3];
  //  double para[10];
  //double tx[3] = {-5., 20., 45.};
  Handle<ESDigiCollection> digis;
  if ( e.getByToken(digilabel_, digis) ) {
    
    for (ESDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr) {
      
      ESDataFrame dataframe = (*digiItr);
      ESDetId id = dataframe.id();
      
      zside = id.zside();
      plane = id.plane();
      ix = id.six();
      iy = id.siy();
      is = id.strip();
      
      //if (zside==1 && plane==1 && ix==15 && iy==6) continue;       
      if (zside==1 && plane==1 && ix==7 && iy==28) continue;       
      if (zside==1 && plane==1 && ix==24 && iy==9 && is==21) continue;       
      if (zside==-1 && plane==2 && ix==35 && iy==17 && is==23) continue;       
      
      int i = (zside==1)? 0:1;
      int j = plane-1;
      
      for (int k=0; k<dataframe.size(); ++k) 
	adc[k] = dataframe.sample(k).adc();
      
      double status = 0;
      if (adc[1] < 200) status = 1;
      if (fabs(adc[0]) > 10) status = 1;
      if (adc[1] < 0 || adc[2] < 0) status = 1;
      if (adc[0] > adc[1] || adc[0] > adc[2]) status = 1;
      if (adc[2] > adc[1]) status = 1;  
      
      if (int(status) == 0) {

	double A1 = adc[1];
	double A2 = adc[2];
	double DeltaT = 25.;
	double aaa = (A2 > 0 && A1 > 0) ? log(A2/A1)/n_ : 20.; // if A1=0, t0=20
	double bbb = wc_/n_*DeltaT;
	double ccc= exp(aaa+bbb);

	double t0 = (2.-ccc)/(1.-ccc) * DeltaT - 5;
	hTiming_[i][j]->Fill(t0);
	//cout<<"t0 : "<<t0<<endl;
	/*
	TGraph *gr = new TGraph(3, tx, adc);
	fit_->SetParameters(50, 10, wc_, n_);
	fit_->FixParameter(2, wc_);
	fit_->FixParameter(3, n_);
	fit_->Print();
	gr->Fit("fitShape", "MQ");
	fit_->GetParameters(para); 
	delete gr;
	//hTiming_[i][j]->Fill(para[1]);
	*/
	//cout<<"ADC : "<<zside<<" "<<plane<<" "<<ix<<" "<<iy<<" "<<is<<" "<<adc[0]<<" "<<adc[1]<<" "<<adc[2]<<" "<<para[1]<<" "<<wc_<<" "<<n_<<endl;

	if (zside == 1) htESP_->Fill(t0);
	else if (zside == -1) htESM_->Fill(t0);
      }
      
    }
  } else {
    LogWarning("ESTimingTask") << "DigiCollection not available";
  }
  
  if (htESP_->GetEntries() > 0 && htESM_->GetEntries() > 0)
    h2DTiming_->Fill(htESM_->GetMean(), htESP_->GetMean());
  
}

void ESTimingTask::set(const edm::EventSetup& es) {

 
  es.get<ESGainRcd>().get(esgain_);
  const ESGain *gain = esgain_.product();
  
  int ESGain = (int) gain->getESGain();

  if (ESGain == 1) { // LG
    wc_ = 0.0837264;
    n_  = 2.016;
  } else { // HG
    wc_ = 0.07291;
    n_  = 1.798; 
  }

  //cout<<"gain : "<<ESGain<<endl;
  //cout<<wc_<<" "<<n_<<endl;
 
}

//define this as a plug-in
DEFINE_FWK_MODULE(ESTimingTask);
