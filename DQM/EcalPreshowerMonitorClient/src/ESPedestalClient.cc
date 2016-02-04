#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalPreshowerMonitorClient/interface/ESPedestalClient.h"

#include <TH1F.h>

using namespace edm;
using namespace std;

ESPedestalClient::ESPedestalClient(const edm::ParameterSet& ps) {
  
  verbose_       = ps.getUntrackedParameter<bool>("verbose", true);
  debug_         = ps.getUntrackedParameter<bool>("debug", true);
  prefixME_	  = ps.getUntrackedParameter<string>("prefixME", "EcalPreshower");
  lookup_	  = ps.getUntrackedParameter<FileInPath>("LookupTable");
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);
  fitPedestal_   = ps.getUntrackedParameter<bool>("fitPedestal", false);
  
  for (int i=0; i<2; i++) 
    for (int j=0; j<2; j++) 
      for (int k=0; k<40; k++) 
	for (int m=0; m<40; m++) {
	  hPed_[i][j][k][m] = 0;
	  hTotN_[i][j][k][m] = 0;
	}
  
}

ESPedestalClient::~ESPedestalClient() {
}

void ESPedestalClient::beginJob(DQMStore* dqmStore) {

   dqmStore_ = dqmStore;

   if ( debug_ ) cout << "ESPedestalClient: beginJob" << endl;

   ievt_ = 0;
   jevt_ = 0;

}

void ESPedestalClient::beginRun(void) {

   if ( debug_ ) cout << "ESPedestalClient: beginRun" << endl;

   jevt_ = 0;

   this->setup();
}

void ESPedestalClient::endJob(void) {

   if ( debug_ ) cout << "ESPedestalClient: endJob, ievt = " << ievt_ << endl;

   // Preform pedestal fit
   char hname[300];
   int iz = 0;
   if (fitPedestal_) {

      if ( verbose_ ) cout<<"ESPedestalClient: Fit Pedestal"<<endl;

      for (int i=0; i<nLines_; ++i) {

	 iz = (senZ_[i]==1) ? 0:1; 

	 for (int is=0; is<32; ++is) {

	    string dirname = prefixME_ + "/ESPedestalTask/";
	    sprintf(hname, "ADC Z %d P %d X %d Y %d Str %d", senZ_[i], senP_[i], senX_[i], senY_[i], is+1);
	    MonitorElement *meFit = dqmStore_->get(dirname+hname);

	    if (meFit==0) continue;
	    TH1F *rootHisto = meFit->getTH1F();
	    rootHisto->Fit("fg", "Q", "", 500, 1800);
	    rootHisto->Fit("fg", "RQ", "", fg->GetParameter(1)-2.*fg->GetParameter(2),fg->GetParameter(1)+2.*fg->GetParameter(2));
	    hPed_[iz][senP_[i]-1][senX_[i]-1][senY_[i]-1]->setBinContent(is+1, (int)(fg->GetParameter(1)+0.5));
	    hTotN_[iz][senP_[i]-1][senX_[i]-1][senY_[i]-1]->setBinContent(is+1, fg->GetParameter(2));

	 }
      } 

   } else {

      if ( verbose_ ) cout<<"ESPedestalClient: Use Histogram Mean"<<endl;

      for (int i=0; i<nLines_; ++i) {

	 iz = (senZ_[i]==1) ? 0:1; 

	 for (int is=0; is<32; ++is) {

	    string dirname = prefixME_ + "/ESPedestalTask/";
	    sprintf(hname, "ADC Z %d P %d X %d Y %d Str %d", senZ_[i], senP_[i], senX_[i], senY_[i], is+1);
	    MonitorElement *meMean = dqmStore_->get(dirname+hname);
	    
	    if (meMean==0) continue;
	    TH1F *rootHisto = meMean->getTH1F();

	    hPed_[iz][senP_[i]-1][senX_[i]-1][senY_[i]-1]->setBinContent(is+1, (int)(rootHisto->GetMean()+0.5));
	    hTotN_[iz][senP_[i]-1][senX_[i]-1][senY_[i]-1]->setBinContent(is+1, rootHisto->GetRMS());

	 } 
      }
   }

   this->cleanup();
}

void ESPedestalClient::endRun(void) {

  if ( debug_ ) cout << "ESPedestalClient: endRun, jevt = " << jevt_ << endl;
  
  this->cleanup();
}

void ESPedestalClient::setup(void) {

   // read in look-up table
   int iz, ip, ix, iy, fed, kchip, pace, bundle, fiber, optorx;
   ifstream file;

   file.open(lookup_.fullPath().c_str());
   if( file.is_open() ) {

      file >> nLines_;

      for (int i=0; i<nLines_; ++i) {
	 file>> iz >> ip >> ix >> iy >> fed >> kchip >> pace >> bundle >> fiber >> optorx;

	 senZ_[i] = iz;
	 senP_[i] = ip;
	 senX_[i] = ix;
	 senY_[i] = iy;
      }

   } else {
      cout<<"ESPedestalClient : Look up table file can not be found in "<<lookup_.fullPath().c_str()<<endl;
   }

   // define histograms
   dqmStore_->setCurrentFolder(prefixME_+"/ESPedestalClient");

   char hname[300];
   for (int i=0; i<nLines_; ++i) {

      iz = (senZ_[i]==1) ? 0:1;

      sprintf(hname, "Ped Z %d P %d X %d Y %d", senZ_[i], senP_[i], senX_[i], senY_[i]);
      hPed_[iz][senP_[i]-1][senX_[i]-1][senY_[i]-1] = dqmStore_->book1D(hname, hname, 32, 0, 32);
      
      sprintf(hname, "Total Noise Z %d P %d X %d Y %d", senZ_[i], senP_[i], senX_[i], senY_[i]);
      hTotN_[iz][senP_[i]-1][senX_[i]-1][senY_[i]-1] = dqmStore_->book1D(hname, hname, 32, 0, 32);
   }

   fg = new TF1("fg", "gaus");

}

void ESPedestalClient::cleanup(void) {

   if( ! enableCleanup_ ) return;

   if ( debug_ ) cout << "ESPedestalClient: cleanup" << endl;

   for (int i=0; i<2; i++)
      for (int j=0; j<2; j++)
	 for (int k=0; k<40; k++)
	    for (int m=0; m<40; m++) {
	       hPed_[i][j][k][m] = 0;
	       hTotN_[i][j][k][m] = 0;
	    }

}

void ESPedestalClient::analyze() {

   ievt_++;
   jevt_++;

}
