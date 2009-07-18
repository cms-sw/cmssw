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
   me = dqmStore_->book2D(histo, histo, 80, 0., 80., 80, 0., 80);
   for ( int i = 0; i < 80; i++ ) {
      for ( int j = 0; j < 80; j++ ) {
	 me->setBinContent( i+1, j+1, 0. );
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

   float DCC[80][80];
   float KChip[80][80];
   float DigiHit[80][80];
   bool noDigiHit = 1;

   MonitorElement* me;

   for (int i=0 ; i<2; ++i){
      for (int j=0 ; j<2; ++j){
	 int iz = (i==0)? 1:-1;
	 sprintf(histo, "ES Integrity Summary 1 Z %d P %d", iz, j+1);
	 me = dqmStore_->get(prefixME_ + "/ESIntegrityClient/" + histo);
	 if(me){
	    for(int x=0; x<40; x++){
	       for(int y=0; y<40; y++){
		  DCC[i*40+x][j*40+y]=me->getBinContent(x+1,y+1);
	       }
	    }
	 }

	 sprintf(histo, "ES Integrity Summary 2 Z %d P %d", iz, j+1);
	 me = dqmStore_->get(prefixME_ + "/ESIntegrityClient/" + histo);
	 if(me){
	    for(int x=0; x<40; x++){
	       for(int y=0; y<40; y++){
		  KChip[i*40+x][j*40+y]=me->getBinContent(x+1,y+1);
	       }
	    }
	 }

	 if(iz==1)  sprintf(histo, "ES+ P%d DigiHit 2D Occupancy", j+1);
	 if(iz==-1) sprintf(histo, "ES- P%d DigiHit 2D Occupancy", j+1);
	 me = dqmStore_->get(prefixME_ + "/ESIntegrityClient/" + histo);
	 if(me){
	    noDigiHit = 0;
	    for(int x=0; x<40; x++){
	       for(int y=0; y<40; y++){
		  DigiHit[i*40+x][j*40+y]=me->getBinContent(x+1,y+1);
	       }
	    }
	 }
      }
   }

   //The global-summary
   int nGlobalErrors = 0;
   int nValidChannels = 0;
   me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryMap");
   if(me){
      float xval = 0;
      for(int x=0; x<80; x++){
	 for(int y=0; y<80; y++){
	    //Number:Error mapping; 1:Normal 2:Noisy 3:No Readout 4:DCC & KChip Error 
	    xval = 1;
	    if(DigiHit[x][y]>1&&noDigiHit==0) xval = 2;
	    if(DigiHit[x][y]<0.01&&noDigiHit==0) xval = 3;
	    if(DCC[x][y]<2.5||DCC[x][y]>3.5) xval = 4;
	    if(KChip[x][y]<2.5||KChip[x][y]>3.5) xval = 4;
	    if(DCC[x][y]<0.5) xval = 0;	//0 for non-module point

	    //Count Valid Channels and Error Channels
	    if( xval != 0 ){
	       nValidChannels++;
	       if( xval != 1) nGlobalErrors++;
	    }

	    me->setBinContent(x+1, y+1, xval);
	 }
      }
   }

   //Return ratio of good channels
   float reportSummary = -1.0;
   if ( nValidChannels != 0 ){
      reportSummary = 1.0 - float(nGlobalErrors)/float(nValidChannels);
   }
   me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummary");
   if ( me ) me->Fill(reportSummary);


}

void ESSummaryClient::softReset(bool flag) {

}


