#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalPreshowerMonitorClient/interface/ESIntegrityClient.h"

using namespace edm;
using namespace std;

ESIntegrityClient::ESIntegrityClient(const ParameterSet& ps) {

  cloneME_       = ps.getUntrackedParameter<bool>("cloneME", true);
  verbose_       = ps.getUntrackedParameter<bool>("verbose", true);
  debug_         = ps.getUntrackedParameter<bool>("debug", false);
  prefixME_      = ps.getUntrackedParameter<string>("prefixME", "");
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);
  lookup_        = ps.getUntrackedParameter<FileInPath>("LookupTable", edm::FileInPath("EventFilter/ESDigiToRaw/data/ES_lookup_table.dat"));

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
    syncStatus_[i] = -1;
    slinkCRCStatus_[i] = -1;
    for (int j=0; j<36; ++j)
      fiberStatus_[i][j] = -1;
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
      fiber_[z-1][ip-1][ix-1][iy-1] = (fiber-1)+(optorx-1)*12;
    }
  } 
  else {
    cout<<"ESIntegrityClient : Look up table file can not be found in "<<lookup_.fullPath().c_str()<<endl;
  }

  hFED_ = 0;
  hFiberOff_ = 0;
  hFiberBadStatus_ = 0;
  hKF1_ = 0;
  hKF2_ = 0;
  hKBC_ = 0;
  hKEC_ = 0; 
  hL1ADiff_ = 0;
  hBXDiff_ = 0;
  hOrbitNumberDiff_ = 0;
  hSLinkCRCErr_ = 0; 

  ievt_ = 0;
  jevt_ = 0;
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
      snprintf(histo, 200, "ES Integrity Summary 1 Z %d P %d", iz, j+1);
      meFED_[i][j] = dqmStore_->book2D(histo, histo, 40, 0.5, 40.5, 40, 0.5, 40.5);
      meFED_[i][j]->setAxisTitle("Si X", 1);
      meFED_[i][j]->setAxisTitle("Si Y", 2);
      
      snprintf(histo, 200, "ES Integrity Summary 2 Z %d P %d", iz, j+1);
      meKCHIP_[i][j] = dqmStore_->book2D(histo, histo, 40, 0.5, 40.5, 40, 0.5, 40.5);
      meKCHIP_[i][j]->setAxisTitle("Si X", 1);
      meKCHIP_[i][j]->setAxisTitle("Si Y", 2);
    }
}

void ESIntegrityClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

}

void ESIntegrityClient::analyze(void) {

  double nDI_FedErr[56];
  for (int i=0; i<56; ++i) nDI_FedErr[i] = 0;

  MonitorElement* me = 0;
  
  me = dqmStore_->get(prefixME_ + "/ESIntegrityTask/ES FEDs used for data taking");
  hFED_ = getHisto(me, cloneME_, hFED_);

  me = dqmStore_->get(prefixME_ + "/ESIntegrityTask/ES Fiber Off");
  hFiberOff_ = getHisto(me, cloneME_, hFiberOff_);

  me = dqmStore_->get(prefixME_ + "/ESIntegrityTask/ES Fiber Bad Status");
  hFiberBadStatus_ = getHisto(me, cloneME_, hFiberBadStatus_);

  me = dqmStore_->get(prefixME_ + "/ESIntegrityTask/ES SLink CRC Errors");
  hSLinkCRCErr_ = getHisto(me, cloneME_, hSLinkCRCErr_); 

  int xval = 0;
  int nevFEDs = 0;
  if (hFED_) {
    for (int i=1; i<=56; ++i) 
      if (nevFEDs < hFED_->GetBinContent(i))
	nevFEDs = (int) hFED_->GetBinContent(i);
  }

  // FED integrity
  for (int i=1; i<=56; ++i) {
    if (hFED_) {
      if (hFED_->GetBinContent(i) > 0) 
	fedStatus_[i-1] = 1;      
    }
    if (hSLinkCRCErr_) {
      if (hSLinkCRCErr_->GetBinContent(i) > 0) 
	slinkCRCStatus_[i-1] = 1;
    }
    for (int j=0; j<36; ++j) {
      if (hFiberBadStatus_) {
	if (hFiberBadStatus_->GetBinContent(i, j+1) > 0) 
	  fiberStatus_[i-1][j] = 2; // bad
	else 
	  fiberStatus_[i-1][j] = 1; // good
      }
      if (hFiberOff_) 
	if (hFiberOff_->GetBinContent(i, j+1) > 0) {
	  fiberStatus_[i-1][j] = 0; // off
	}
    }
  }

  // obtain MEs from ESRawDataTask
  me = dqmStore_->get(prefixME_ + "/ESRawDataTask/ES L1A DCC errors");
  hL1ADiff_ = getHisto(me, cloneME_, hL1ADiff_); 

  me = dqmStore_->get(prefixME_ + "/ESRawDataTask/ES BX DCC errors");
  hBXDiff_ = getHisto(me, cloneME_, hBXDiff_); 

  me = dqmStore_->get(prefixME_ + "/ESRawDataTask/ES Orbit Number DCC errors");
  hOrbitNumberDiff_ = getHisto(me, cloneME_, hOrbitNumberDiff_); 

  for (int i=1; i<=56; ++i) {
    if (hL1ADiff_ && hBXDiff_) {
      if (hL1ADiff_->GetBinContent(i) > 0 || hBXDiff_->GetBinContent(i) > 0) {
	syncStatus_[i-1] = 1;
	if (hL1ADiff_->GetBinContent(i) > nDI_FedErr[i-1]) nDI_FedErr[i-1] = hL1ADiff_->GetBinContent(i);
	if (hBXDiff_->GetBinContent(i) > nDI_FedErr[i-1]) nDI_FedErr[i-1] = hBXDiff_->GetBinContent(i);
      }
    }
    //if (hOrbitNumberDiff_->GetBinContent(i) > 0) syncStatus_[i-1] = 1;
  }

  // KCHIP integrity
  me = dqmStore_->get(prefixME_ + "/ESIntegrityTask/ES KChip Flag 1 Error codes");
  hKF1_ = getHisto(me, cloneME_, hKF1_); 

  me = dqmStore_->get(prefixME_ + "/ESIntegrityTask/ES KChip Flag 2 Error codes");
  hKF2_ = getHisto(me, cloneME_, hKF2_); 

  me = dqmStore_->get(prefixME_ + "/ESIntegrityTask/ES KChip BC mismatch with OptoRX");
  hKBC_ = getHisto(me, cloneME_, hKBC_); 

  me = dqmStore_->get(prefixME_ + "/ESIntegrityTask/ES KChip EC mismatch with OptoRX");
  hKEC_ = getHisto(me, cloneME_, hKEC_); 

  Int_t kchip_xval[1550];
  for (int i=0; i<1550; ++i) {

    xval = 3;
    Int_t kErr = 0;
    for (int j=1; j<16; ++j) { 
      if (hKF1_) {
	if (hKF1_->GetBinContent(i, j+1)>0) {
	  xval = 2;
	  kErr++;
	}
      }
      if (hKF2_) {
	if (hKF2_->GetBinContent(i, j+1)>0) {
	  xval = 4;
	  kErr++;
	}
      }
    }
    if (hKBC_) {
      if (hKBC_->GetBinContent(i)>0) {
	xval = 5;
	kErr++;
      } 
    }
    if (hKEC_) {
      if (hKEC_->GetBinContent(i)>0) {
	xval = 6;
	kErr++;
      } 
    }
    if (kErr>1) xval = 7; 
    kchip_xval[i] = xval;
  }

  // detailed KCHIP summary (i.e. summary 2)
  for (int iz=0; iz<2; ++iz) 
    for (int ip=0; ip<2; ++ip)
      for (int ix=0; ix<40; ++ix)
	for (int iy=0; iy<40; ++iy) {
	  if (fed_[iz][ip][ix][iy] == -1) continue;
	  if (fedStatus_[fed_[iz][ip][ix][iy]-520]==-1 || fiberStatus_[fed_[iz][ip][ix][iy]-520][fiber_[iz][ip][ix][iy]] == 0)
	    kchip_xval[kchip_[iz][ip][ix][iy]-1] = 0;
	  if ((kchip_[iz][ip][ix][iy]-2) >= 0) 
	    meKCHIP_[iz][ip]->setBinContent(ix+1, iy+1, kchip_xval[kchip_[iz][ip][ix][iy]-2]); 
	}

  // FED, Fiber, KCHIP summary (i.e. summary 1) 
  Int_t nErr = 0;
  for (int iz=0; iz<2; ++iz) 
    for (int ip=0; ip<2; ++ip)
      for (int ix=0; ix<40; ++ix)
	for (int iy=0; iy<40; ++iy) {

	  if (fed_[iz][ip][ix][iy] == -1) continue;
	  nErr = 0;

	  if (fedStatus_[fed_[iz][ip][ix][iy]-520] == 1) {

	    if (hFED_) {
	      if (hFED_->GetBinContent(fed_[iz][ip][ix][iy]-520+1) == nevFEDs) 
		xval = 3;
	      else {
		xval = 4; // FED problem
		nErr++;
	      }
	    }

	    if (fiberStatus_[fed_[iz][ip][ix][iy]-520][fiber_[iz][ip][ix][iy]] == 2) {
	      xval = 2; // fiber problem
	      nErr++;
	    }
	    if (fiberStatus_[fed_[iz][ip][ix][iy]-520][fiber_[iz][ip][ix][iy]] == 0) {
	      xval = 0; // fiber off
	    } 
	    if (kchip_xval[kchip_[iz][ip][ix][iy]-1] != 3 && kchip_xval[kchip_[iz][ip][ix][iy]-1] != 0) {
	      xval = 5; // kchip problem
	      nErr++;
	    }
	    if (syncStatus_[fed_[iz][ip][ix][iy]-520] == 1) {
	      xval = 6;
	      nErr++;
	    }
	    if (slinkCRCStatus_[fed_[iz][ip][ix][iy]-520] == 1) {
	      xval = 8;
	      nErr++;
	    }
	    if (nErr > 1) xval = 7;
	  } else if (fedStatus_[fed_[iz][ip][ix][iy]-520] == -1) {
	    xval = 0;
	  } 
	  
	  meFED_[iz][ip]->setBinContent(ix+1, iy+1, xval); 
	}
  
}

void ESIntegrityClient::softReset(bool flag) {
  
}


