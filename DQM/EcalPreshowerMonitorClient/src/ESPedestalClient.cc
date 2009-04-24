#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>


#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"


#include "DQM/EcalPreshowerMonitorClient/interface/ESPedestalClient.h"

#include <TH1F.h>


using namespace cms;
using namespace edm;
using namespace std;

ESPedestalClient::ESPedestalClient(const edm::ParameterSet& ps){


	prefixME_	= ps.getUntrackedParameter<string>("prefixME", "EcalPreshower");
	lookup_		= ps.getUntrackedParameter<FileInPath>("LookupTable");
	enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);




	for (int i=0; i<2; i++) 
		for (int j=0; j<2; j++) 
			for (int k=0; k<40; k++) 
				for (int m=0; m<40; m++){
					hPed_[i][j][k][m] = 0;
					hTotN_[i][j][k][m] = 0;
				}


}


ESPedestalClient::~ESPedestalClient(){

}

void ESPedestalClient::beginJob(DQMStore* dqmStore) {

	dqmStore_ = dqmStore;

	EvtperRun_ = 0;
	EvtperJob_ = 0;

}

void ESPedestalClient::beginRun(void) {

	EvtperRun_ = 0;

	this->setup();

}

void ESPedestalClient::endJob(void) {

	this->cleanup();

}

void ESPedestalClient::endRun(void) {

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
		cout<<"ESUnpackerV4::ESUnpackerV4 : Look up table file can not be found in "<<lookup_.fullPath().c_str()<<endl;
	}

	// define histograms
	dqmStore_->setCurrentFolder(prefixME_+"/ESPedestalClient");

	Char_t hname[300];
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

	for (int i=0; i<2; i++)
		for (int j=0; j<2; j++)
			for (int k=0; k<40; k++)
				for (int m=0; m<40; m++){
					hPed_[i][j][k][m] = 0;
					hTotN_[i][j][k][m] = 0;
				}



}

void ESPedestalClient::analyze()
{
	EvtperJob_++;
	EvtperRun_++;

	// Preform pedestal fit
	char hname[300];
	int iz = 0;
	for (int i=0; i<nLines_; ++i) {
		iz = (senZ_[i]==1) ? 0:1; 
		for (int is=0; is<32; ++is) {

			string dirname = prefixME_ + "/ESPedestalTask/";
			sprintf(hname, "ADC Z %d P %d X %d Y %d Str %d", senZ_[i], senP_[i], senX_[i], senY_[i], is+1);
			MonitorElement * meHisto = dqmStore_->get(dirname+hname);

			if (meHisto==0) continue;
			TH1F *rootHisto = meHisto->getTH1F();
			rootHisto->Fit("fg", "Q", "", 500, 1800);
			rootHisto->Fit("fg", "RQ", "", fg->GetParameter(1)-2.*fg->GetParameter(2),fg->GetParameter(1)+2.*fg->GetParameter(2));

			hPed_[iz][senP_[i]-1][senX_[i]-1][senY_[i]-1]->setBinContent(is+1, (int)(fg->GetParameter(1)+0.5));
			hTotN_[iz][senP_[i]-1][senX_[i]-1][senY_[i]-1]->setBinContent(is+1, fg->GetParameter(2));
		} 
	}
}

