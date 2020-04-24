#include "DQM/EcalPreshowerMonitorClient/interface/ESPedestalClient.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

ESPedestalClient::ESPedestalClient(const edm::ParameterSet& ps) :
  ESClient(ps),
  fitPedestal_(ps.getUntrackedParameter<bool>("fitPedestal")),
  fg_(new TF1("fg", "gaus")),
  senZ_(),
  senP_(),
  senX_(),
  senY_()
{
  for (int i=0; i<2; i++) 
    for (int j=0; j<2; j++) 
      for (int k=0; k<40; k++) 
	for (int m=0; m<40; m++) {
	  hPed_[i][j][k][m] = nullptr;
	  hTotN_[i][j][k][m] = nullptr;
	}

   std::string lutPath(ps.getUntrackedParameter<edm::FileInPath>("LookupTable").fullPath());

   // read in look-up table
   int nLines, iz, ip, ix, iy, fed, kchip, pace, bundle, fiber, optorx;
   ifstream file(lutPath);

   if( file.is_open() ) {

      file >> nLines;

      for (int i=0; i<nLines; ++i) {
	 file>> iz >> ip >> ix >> iy >> fed >> kchip >> pace >> bundle >> fiber >> optorx;

	 senZ_.push_back(iz);
	 senP_.push_back(ip);
	 senX_.push_back(ix);
	 senY_.push_back(iy);
      }

      file.close();
   } else {
      cout<<"ESPedestalClient : Look up table file can not be found in "<<lutPath<<endl;
   }
}

ESPedestalClient::~ESPedestalClient() {
  delete fg_;
}

void ESPedestalClient::endJobAnalyze(DQMStore::IGetter& _igetter) {

   if ( debug_ ) cout << "ESPedestalClient: endJob" << endl;

   // Preform pedestal fit
   char hname[300];
   int iz = 0;
   if (fitPedestal_) {

      if ( verbose_ ) cout<<"ESPedestalClient: Fit Pedestal"<<endl;

      for (unsigned i=0; i<senZ_.size(); ++i) {

	 iz = (senZ_[i]==1) ? 0:1; 

	 for (int is=0; is<32; ++is) {

	    string dirname = prefixME_ + "/ESPedestalTask/";
	    sprintf(hname, "ADC Z %d P %d X %d Y %d Str %d", senZ_[i], senP_[i], senX_[i], senY_[i], is+1);
	    MonitorElement *meFit = _igetter.get(dirname+hname);

	    if (meFit==nullptr) continue;
	    TH1F *rootHisto = meFit->getTH1F();

	    rootHisto->Fit(fg_, "Q", "", 500, 1800);
	    rootHisto->Fit(fg_, "RQ", "", fg_->GetParameter(1)-2.*fg_->GetParameter(2),fg_->GetParameter(1)+2.*fg_->GetParameter(2));
	    hPed_[iz][senP_[i]-1][senX_[i]-1][senY_[i]-1]->setBinContent(is+1, (int)(fg_->GetParameter(1)+0.5));
	    hTotN_[iz][senP_[i]-1][senX_[i]-1][senY_[i]-1]->setBinContent(is+1, fg_->GetParameter(2));
	 }
      } 

   } else {

      if ( verbose_ ) cout<<"ESPedestalClient: Use Histogram Mean"<<endl;

      for (unsigned i=0; i<senZ_.size(); ++i) {

	 iz = (senZ_[i]==1) ? 0:1; 

	 for (int is=0; is<32; ++is) {

	    string dirname = prefixME_ + "/ESPedestalTask/";
	    sprintf(hname, "ADC Z %d P %d X %d Y %d Str %d", senZ_[i], senP_[i], senX_[i], senY_[i], is+1);
	    MonitorElement *meMean = _igetter.get(dirname+hname);
	    
	    if (meMean==nullptr) continue;
	    TH1F *rootHisto = meMean->getTH1F();

	    hPed_[iz][senP_[i]-1][senX_[i]-1][senY_[i]-1]->setBinContent(is+1, (int)(rootHisto->GetMean()+0.5));
	    hTotN_[iz][senP_[i]-1][senX_[i]-1][senY_[i]-1]->setBinContent(is+1, rootHisto->GetRMS());

	 } 
      }
   }
}

void ESPedestalClient::book(DQMStore::IBooker& _ibooker) {

   // define histograms
   _ibooker.setCurrentFolder(prefixME_+"/ESPedestalClient");

   char hname[300];
   for (unsigned i=0; i<senZ_.size(); ++i) {

     int iz = (senZ_[i]==1) ? 0:1;

      sprintf(hname, "Ped Z %d P %d X %d Y %d", senZ_[i], senP_[i], senX_[i], senY_[i]);
      hPed_[iz][senP_[i]-1][senX_[i]-1][senY_[i]-1] = _ibooker.book1D(hname, hname, 32, 0, 32);
      
      sprintf(hname, "Total Noise Z %d P %d X %d Y %d", senZ_[i], senP_[i], senX_[i], senY_[i]);
      hTotN_[iz][senP_[i]-1][senX_[i]-1][senY_[i]-1] = _ibooker.book1D(hname, hname, 32, 0, 32);
   }

}
