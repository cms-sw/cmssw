#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalPreshowerMonitorClient/interface/ESSummaryClient.h"

using namespace cms;
using namespace edm;
using namespace std;

ESSummaryClient::ESSummaryClient(const ParameterSet& ps) {

   cloneME_       = ps.getUntrackedParameter<bool>("cloneME", true);
   verbose_       = ps.getUntrackedParameter<bool>("verbose", true);
   debug_         = ps.getUntrackedParameter<bool>("debug", false);
   prefixME_      = ps.getUntrackedParameter<string>("prefixME", "");
   enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

}

ESSummaryClient::~ESSummaryClient() {
}

void ESSummaryClient::beginJob(DQMStore* dqmStore) {

   dqmStore_ = dqmStore;

   if ( debug_ ) cout << "ESSummaryClient: beginJob" << endl;

   ievt_ = 0;
   jevt_ = 0;

   char histo[200];

   MonitorElement* me;

   dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo" );

   sprintf(histo, "reportSummary");
   me = dqmStore_->get(prefixME_ + "/EventInfo/" + histo);
   if ( me ) {
      dqmStore_->removeElement(me->getName());
   }
   me = dqmStore_->bookFloat(histo);
   me->Fill(-1.0);      


   dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo" );

   sprintf(histo, "reportSummaryMap");
   me = dqmStore_->get(prefixME_ + "/EventInfo/" + histo);
   if ( me ) {
      dqmStore_->removeElement(me->getName());
   }
   me = dqmStore_->book2D(histo, histo, 80, 0.5, 80.5, 80, 0.5, 80.5);
   me->setAxisTitle("Si X", 1);
   me->setAxisTitle("Si Y", 2);

   for ( int i = 0; i < 80; i++ ) {
      for ( int j = 0; j < 80; j++ ) {
	 me->setBinContent( i+1, j+1, -1. );
      }
   }

}

void ESSummaryClient::beginRun(void) {

   if ( debug_ ) cout << "ESSummaryClient: beginRun" << endl;

   jevt_ = 0;

   this->setup();

}

void ESSummaryClient::endJob(void) {

   if ( debug_ ) cout << "ESSummaryClient: endJob, ievt = " << ievt_ << endl;

   this->cleanup();

}

void ESSummaryClient::endRun(void) {

   if ( debug_ ) cout << "ESSummaryClient: endRun, jevt = " << jevt_ << endl;

   this->cleanup();

}

void ESSummaryClient::setup(void) {

}

void ESSummaryClient::cleanup(void) {

   if ( ! enableCleanup_ ) return;

}

void ESSummaryClient::analyze(void) {

   char histo[200];

   float nDI_FedErr[80][80];
   float DCC[80][80];
   float eCount;

   MonitorElement* me;

   for ( int i = 0; i < 80; i++) {
      for( int j = 0; j < 80; j++) {
	 nDI_FedErr[i][j] = 0;
	 DCC[i][j]=0;
      }
   }

   for (int i=0 ; i<2; ++i){
      for (int j=0 ; j<2; ++j){
	 int iz = (i==0)? 1:-1;
	 sprintf(histo, "ES Integrity Errors 1 Z %d P %d", iz, j+1);
	 me = dqmStore_->get(prefixME_ + "/ESIntegrityClient/" + histo);
	 if(me){
	    for(int x=0; x<40; x++){
	       for(int y=0; y<40; y++){
		  nDI_FedErr[i*40+x][(1-j)*40+y]=me->getBinContent(x+1,y+1);
	       }
	    }
	 }
	 
	 sprintf(histo, "ES Integrity Summary 1 Z %d P %d", iz, j+1);
	 me = dqmStore_->get(prefixME_ + "/ESIntegrityClient/" + histo);
	 if(me){
	    for(int x=0; x<40; x++){
	       for(int y=0; y<40; y++){
		  DCC[i*40+x][(1-j)*40+y]=me->getBinContent(x+1,y+1);
	       }
	    }
	 }

	 sprintf(histo, "ES Digi 2D Occupancy Z %d P %d", iz, j+1);
	 me = dqmStore_->get(prefixME_ + "/ESOccupancyTask/" + histo);
	 if(me){
	    eCount = me->getBinContent(40,40);
	 }
	 else {
	    eCount = 1.;
	 } 
      }
   }

   //The global-summary
   //ReportSummary Map 
   //  ES+F  ES-F
   //  ES+R  ES-R
   float nValidChannels=0; 
   float nGlobalErrors=0;
   me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryMap");
   if(me){
      for(int x=0; x<80; x++){
	 for(int y=0; y<80; y++){
	    if( nDI_FedErr[x][y] >= 0. ){
	       me->setBinContent(x+1, y+1, 1-(nDI_FedErr[x][y]/eCount) );
	       nValidChannels+=1.;
	       nGlobalErrors+=nDI_FedErr[x][y]/eCount;
	    }
	    else {
	       me->setBinContent(x+1, y+1, -1.);
	    }
	    if(DCC[x][y]==0.) me->setBinContent(x+1, y+1, -1.);
	 }
      }
   }

   //Return ratio of good channels
   float reportSummary = -1.0;
   if ( nValidChannels != 0 ){
      reportSummary = 1.0 - nGlobalErrors/nValidChannels;
   }
   me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummary");
   if ( me ) me->Fill(reportSummary);


}

void ESSummaryClient::softReset(bool flag) {

}


