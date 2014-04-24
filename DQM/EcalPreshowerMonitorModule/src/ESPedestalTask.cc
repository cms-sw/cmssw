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
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalPreshowerMonitorModule/interface/ESPedestalTask.h"

using namespace cms;
using namespace edm;
using namespace std;

ESPedestalTask::ESPedestalTask(const edm::ParameterSet& ps) {
  
  digitoken_ 	= consumes<ESDigiCollection>(ps.getParameter<InputTag>("DigiLabel"));
  lookup_     	= ps.getUntrackedParameter<FileInPath>("LookupTable");
  outputFile_ 	= ps.getUntrackedParameter<string>("OutputFile","");
  prefixME_	= ps.getUntrackedParameter<string>("prefixME", "EcalPreshower"); 

  for (int i=0; i<2; ++i) 
    for (int j=0; j<2; ++j) 
      for (int k=0; k<40; ++k)
	for (int l=0; l<40; ++l)
	  senCount_[i][j][k][l] = -1;

  ievt_ = 0;
}

void ESPedestalTask::bookHistograms(DQMStore::IBooker& iBooker, Run const&, EventSetup const&)
{  
  int iz, ip, ix, iy, fed, kchip, pace, bundle, fiber, optorx;
  int senZ_[4288], senP_[4288], senX_[4288], senY_[4288];
  
  // read in look-up table
  ifstream file(lookup_.fullPath().c_str());
  if(!file.is_open())
    throw cms::Exception("FileNotFound") << lookup_.fullPath();
    
  file >> nLines_;
    
  for (int i=0; i<nLines_; ++i) {
    file>> iz >> ip >> ix >> iy >> fed >> kchip >> pace >> bundle >> fiber >> optorx;
      
    senZ_[i] = iz;
    senP_[i] = ip;
    senX_[i] = ix;
    senY_[i] = iy;

    iz = (senZ_[i]==1) ? 0:1;
    senCount_[iz][senP_[i]-1][senX_[i]-1][senY_[i]-1] = i; 
  }
 
  char hname[300];
  
  iBooker.setCurrentFolder(prefixME_ + "/ESPedestalTask");
    
  for (int i=0; i<nLines_; ++i) {
    for (int is=0; is<32; ++is) {
      sprintf(hname, "ADC Z %d P %d X %d Y %d Str %d", senZ_[i], senP_[i], senX_[i], senY_[i], is+1);
      meADC_[i][is] = iBooker.book1D(hname, hname, 1000, 899.5, 1899.5);
    }
  }
}

void ESPedestalTask::endJob(void) {

  LogInfo("ESPedestalTask") << "analyzed " << ievt_ << " events";

}

void ESPedestalTask::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  
  ievt_++;  
  runNum_ = e.id().run();
  
  Handle<ESDigiCollection> digis;
  e.getByToken(digitoken_, digis);
  
  runtype_ = 1; // Let runtype_ = 1
  
  // Digis
  int zside, plane, ix, iy, strip, iz;
  if (digis.isValid()) {
    for (ESDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr) {
      
      ESDataFrame dataframe = (*digiItr);
      ESDetId id = dataframe.id();
      
      zside = id.zside();
      plane = id.plane();
      ix    = id.six();
      iy    = id.siy();
      strip = id.strip();
      iz = (zside==1) ? 0:1;
      
      if (meADC_[senCount_[iz][plane-1][ix-1][iy-1]][strip-1]) {
	if(runtype_ == 1){		
	  meADC_[senCount_[iz][plane-1][ix-1][iy-1]][strip-1]->Fill(dataframe.sample(0).adc());
	  meADC_[senCount_[iz][plane-1][ix-1][iy-1]][strip-1]->Fill(dataframe.sample(1).adc());
	  meADC_[senCount_[iz][plane-1][ix-1][iy-1]][strip-1]->Fill(dataframe.sample(2).adc());
	} else if(runtype_ == 3) {
	  meADC_[senCount_[iz][plane-1][ix-1][iy-1]][strip-1]->Fill(dataframe.sample(1).adc());
	}
      }	
      
    }
  }
 
}

DEFINE_FWK_MODULE(ESPedestalTask);
