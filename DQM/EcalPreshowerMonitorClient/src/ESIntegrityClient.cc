#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalPreshowerMonitorClient/interface/ESIntegrityClient.h"

using namespace cms;
using namespace edm;
using namespace std;

ESIntegrityClient::ESIntegrityClient(const ParameterSet& ps) {

  cloneME_       = ps.getUntrackedParameter<bool>("cloneME", true);
  verbose_       = ps.getUntrackedParameter<bool>("verbose", true);
  debug_         = ps.getUntrackedParameter<bool>("debug", false);
  prefixME_      = ps.getUntrackedParameter<string>("prefixME", "");
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);
  lookup_        = ps.getUntrackedParameter<FileInPath>("LookupTable");

  // read in look-up table
  for (int i=0; i<2; ++i) 
    for (int j=0; j<2; ++j) 
      for (int k=0; k<40; ++k) 
	for (int m=0; m<40; ++m) {
	  fed_[i][j][k][m] = -1; 
	  kchip_[i][j][k][m] = -1;
	}

  for (int i=0; i<56; ++i) {
    fedStatus_[i] = -1;
    fiberStatus_[i] = -1;
  }

  int nLines_, z, iz, ip, ix, iy, fed, kchip, pace, bundle, fiber, optorx;
  ifstream file;
  file.open(lookup_.fullPath().c_str());
  if( file.is_open() ) {
    
    file >> nLines_;
    
    for (int i=0; i<nLines_; ++i) {
      file>> iz >> ip >> ix >> iy >> fed >> kchip >> pace >> bundle >> fiber >> optorx;
      
      z = (iz==-1) ? 2:iz;
      fed_[z-1][ip-1][ix-1][iy-1] = fed;
      kchip_[z-1][ip-1][ix-1][iy-1] = kchip;
    }
  } 
  else {
    cout<<"ESUnpackerV4::ESUnpackerV4 : Look up table file can not be found in "<<lookup_.fullPath().c_str()<<endl;
  }

}

ESIntegrityClient::~ESIntegrityClient() {
}

void ESIntegrityClient::beginJob(DQMStore* dqmStore) {

  dqmStore_ = dqmStore;

  if ( debug_ ) cout << "ESIntegrityClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void ESIntegrityClient::beginRun(void) {

  if ( debug_ ) cout << "ESIntegrityClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

}

void ESIntegrityClient::endJob(void) {

  if ( debug_ ) cout << "ESIntegrityClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

}

void ESIntegrityClient::endRun(void) {

  if ( debug_ ) cout << "ESIntegrityClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();

}

void ESIntegrityClient::setup(void) {

  char histo[200];

  dqmStore_->setCurrentFolder( prefixME_ + "/ESIntegrityClient" );

  for (int i=0 ; i<2; ++i) 
    for (int j=0 ; j<2; ++j) {
      int iz = (i==0)? 1:-1;
      sprintf(histo, "ES Integrity Summary 1 Z %d P %d", iz, j+1);
      meFED_[i][j] = dqmStore_->book2D(histo, histo, 40, 0.5, 40.5, 40, 0.5, 40.5);
      meFED_[i][j]->setAxisTitle("Si X", 1);
      meFED_[i][j]->setAxisTitle("Si Y", 2);
      
      sprintf(histo, "ES Integrity Summary 2 Z %d P %d", iz, j+1);
      meKCHIP_[i][j] = dqmStore_->book2D(histo, histo, 40, 0.5, 40.5, 40, 0.5, 40.5);
      meKCHIP_[i][j]->setAxisTitle("Si X", 1);
      meKCHIP_[i][j]->setAxisTitle("Si Y", 2);
    }
}

void ESIntegrityClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

}

void ESIntegrityClient::analyze(void) {
  
  char histo[200];
  
  MonitorElement* me;
  
  sprintf(histo, (prefixME_ + "/ESIntegrityTask/ES FEDs used for data taking").c_str());
  me = dqmStore_->get(histo);
  hFED_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, hFED_ ); 

  sprintf(histo, (prefixME_ + "/ESIntegrityTask/ES Fiber Status").c_str());
  me = dqmStore_->get(histo);
  hFiber_ = UtilsClient::getHisto<TH2F*>( me, cloneME_, hFiber_ ); 
  
  int xval = 0;
  int nevFEDs = 0;
  for (int i=1; i<=56; ++i) 
    if (nevFEDs < hFED_->GetBinContent(i))
      nevFEDs = (int) hFED_->GetBinContent(i);
  
  // FED integrity
  for (int i=1; i<=56; ++i) {
    if (hFED_->GetBinContent(i) > 0) 
      fedStatus_[i-1] = 1;      
    if (hFiber_->GetBinContent(i+58) > 0) 
      fiberStatus_[i-1] = 1;
  }

  for (int iz=0; iz<2; ++iz) 
    for (int ip=0; ip<2; ++ip)
      for (int ix=0; ix<40; ++ix)
	for (int iy=0; iy<40; ++iy) {
	  if (fed_[iz][ip][ix][iy] == -1) continue;
	  if (fedStatus_[fed_[iz][ip][ix][iy]-520] == 1) {
	    xval = (hFED_->GetBinContent(fed_[iz][ip][ix][iy]-520+1) == nevFEDs) ? 3:5;
	    if (fiberStatus_[fed_[iz][ip][ix][iy]-520] == 1) xval = 2;
	  } else if (fedStatus_[fed_[iz][ip][ix][iy]-520] == -1) {
	    xval = 0;
	  } 
	  meFED_[iz][ip]->setBinContent(ix+1, iy+1, xval); 
	}
  
  // KCHIP integrity
  sprintf(histo, (prefixME_ + "/ESIntegrityTask/ES KChip Flag 1 Error codes").c_str());
  me = dqmStore_->get(histo);
  hKF1_ = UtilsClient::getHisto<TH2F*>( me, cloneME_, hKF1_ ); 

  sprintf(histo, (prefixME_ + "/ESIntegrityTask/ES KChip Flag 2 Error codes").c_str());
  me = dqmStore_->get(histo);
  hKF2_ = UtilsClient::getHisto<TH2F*>( me, cloneME_, hKF2_ ); 

  sprintf(histo, (prefixME_ + "/ESIntegrityTask/ES KChip BC mismatch with OptoRX").c_str());
  me = dqmStore_->get(histo);
  hKBC_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, hKBC_ ); 

  sprintf(histo, (prefixME_ + "/ESIntegrityTask/ES KChip EC mismatch with OptoRX").c_str());
  me = dqmStore_->get(histo);
  hKEC_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, hKEC_ ); 

  Int_t kchip_xval[1550];
  for (int i=0; i<=1550; ++i) {

    xval = 3;
    Int_t kErr = 0;
    for (int j=1; j<16; ++j) { 
      if (hKF1_->GetBinContent(i, j+1)>0) {
	xval = 2;
	kErr++;
      }
      if (hKF2_->GetBinContent(i, j+1)>0) {
	xval = 4;
	kErr++;
      }
    }
    if (hKBC_->GetBinContent(i)>0) {
      xval = 5;
      kErr++;
    } 
    if (hKEC_->GetBinContent(i)>0) {
      xval = 6;
      kErr++;
    } 
    if (kErr>1) xval = 7; 
    kchip_xval[i] = xval;
  }

  for (int iz=0; iz<2; ++iz) 
    for (int ip=0; ip<2; ++ip)
      for (int ix=0; ix<40; ++ix)
	for (int iy=0; iy<40; ++iy) {
	  if (fed_[iz][ip][ix][iy] == -1) continue;
	  if (fedStatus_[fed_[iz][ip][ix][iy]-520] == -1) kchip_xval[kchip_[iz][ip][ix][iy]-1] = 0;
	  meKCHIP_[iz][ip]->setBinContent(ix+1, iy+1, kchip_xval[kchip_[iz][ip][ix][iy]-1]); 
	}
  
}

void ESIntegrityClient::softReset(bool flag) {
  
}


