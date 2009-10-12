
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
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"


#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalPreshowerMonitorModule/interface/ESPedestalTask.h"


using namespace cms;
using namespace edm;
using namespace std;

ESPedestalTask::ESPedestalTask(const edm::ParameterSet& ps)

{

   digilabel_    	= ps.getParameter<InputTag>("DigiLabel");
   lookup_       	= ps.getUntrackedParameter<FileInPath>("LookupTable");
   outputFile_ 	= ps.getUntrackedParameter<string>("OutputFile","");
   prefixME_	= ps.getUntrackedParameter<string>("prefixME", "EcalPreshower"); 

   dqmStore_	= Service<DQMStore>().operator->();

   int iz, ip, ix, iy, fed, kchip, pace, bundle, fiber, optorx;

   int senZ_[4288], senP_[4288], senX_[4288], senY_[4288];


   // read in look-up table
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
   } 
   else {
      cout<<"ESUnpackerV4::ESUnpackerV4 : Look up table file can not be found in "<<lookup_.fullPath().c_str()<<endl;
   }



   //Histogram init  
   dqmStore_->setCurrentFolder(prefixME_ + "/ESPedestalTask");


   for(int i=0; i<2 ; i++){
      for(int j=0; j<2 ; j++){
	 for(int k=0; k<40 ; k++){	
	    for(int l=0; l<40 ; l++){
	       for(int m=0; m<32 ; m++){
		  hADC_[i][j][k][l][m] = 0;
	       }}}}}


   char hname[300];
   int count = 0;
   for (int i=0; i<nLines_; ++i) {
      iz = (senZ_[i]==1) ? 0:1;
      for (int is=0; is<32; ++is) {
	 sprintf(hname, "ADC Z %d P %d X %d Y %d Str %d", senZ_[i], senP_[i], senX_[i], senY_[i], is+1);
	 hADC_[iz][senP_[i]-1][senX_[i]-1][senY_[i]-1][is] = dqmStore_->book1D(hname, hname, 4100, 0, 4100);
	 dqmStore_->tag(hADC_[iz][senP_[i]-1][senX_[i]-1][senY_[i]-1][is]->getFullname(),++count);
      }
   }
}

ESPedestalTask::~ESPedestalTask() {
}

void ESPedestalTask::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {

   runNum_ = e.id().run();
   eCount_++;

   Handle<ESDigiCollection> digis;
   if ( e.getByLabel(digilabel_, digis) ) {

     runtype_ = 1; // Let runtype_ = 1
     
     // Digis
     int zside, plane, ix, iy, strip, iz;
     for (ESDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr) {
       
       ESDataFrame dataframe = (*digiItr);
       ESDetId id = dataframe.id();
       
       zside = id.zside();
       plane = id.plane();
       ix    = id.six();
       iy    = id.siy();
       strip = id.strip();
       iz = (zside==1) ? 0:1;
       
       layer_ = plane;
       
       
       if(hADC_[iz][plane-1][ix-1][iy-1][strip-1]){
	 if(runtype_ == 1){		
	   hADC_[iz][plane-1][ix-1][iy-1][strip-1]->Fill(dataframe.sample(0).adc());
	   hADC_[iz][plane-1][ix-1][iy-1][strip-1]->Fill(dataframe.sample(1).adc());
	   hADC_[iz][plane-1][ix-1][iy-1][strip-1]->Fill(dataframe.sample(2).adc());
	 } else if(runtype_ == 3) {
	   hADC_[iz][plane-1][ix-1][iy-1][strip-1]->Fill(dataframe.sample(1).adc());
	 }
       }	
     }
   } else {
     LogWarning("ESPedestalTask") << digilabel_ << " not available";
   }     

}

void ESPedestalTask::beginJob(const edm::EventSetup & c) {
}

void ESPedestalTask::endJob() {

   if(outputFile_.size() != 0){
      cout<<"Save result to "<<outputFile_<<endl;
      dqmStore_->save(outputFile_);
   }

}

//define this as a plug-in
DEFINE_FWK_MODULE(ESPedestalTask);
