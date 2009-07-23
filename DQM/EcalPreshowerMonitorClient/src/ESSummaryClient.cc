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

   float DCC[80][80];
   float KChip[80][80];
   float Digi[80][80];
   bool UseDCC, UseKChip, UseDigi;

   MonitorElement* me;

   UseDCC = false;
   UseKChip = false;
   UseDigi = false;

   for ( int i = 0; i < 80; i++) {
      for( int j = 0; j < 80; j++) {
	 DCC[i][j] = 0;
	 KChip[i][j] = 0;
	 Digi[i][j] = 0;
      }
   }

   for (int i=0 ; i<2; ++i){
      for (int j=0 ; j<2; ++j){
	 int iz = (i==0)? 1:-1;
	 sprintf(histo, "ES Integrity Summary 1 Z %d P %d", iz, j+1);
	 me = dqmStore_->get(prefixME_ + "/ESIntegrityClient/" + histo);
	 if(me){
	    UseDCC = true;
	    for(int x=0; x<40; x++){
	       for(int y=0; y<40; y++){
		  DCC[i*40+x][(1-j)*40+y]=me->getBinContent(x+1,y+1);
	       }
	    }
	 }

	 sprintf(histo, "ES Integrity Summary 2 Z %d P %d", iz, j+1);
	 me = dqmStore_->get(prefixME_ + "/ESIntegrityClient/" + histo);
	 if(me){
	    UseKChip = true;
	    for(int x=0; x<40; x++){
	       for(int y=0; y<40; y++){
		  KChip[i*40+x][(1-j)*40+y]=me->getBinContent(x+1,y+1);
	       }
	    }
	 }

	 sprintf(histo, "ES Digi 2D Occupancy Z %d P %d", iz, j+1);
	 me = dqmStore_->get(prefixME_ + "/ESOccupancyTask/" + histo);
	 if(me){
	    UseDigi = true;
	    float eCount = me->getBinContent(40,40);
	    for(int x=0; x<40; x++){
	       for(int y=0; y<40; y++){
		  Digi[i*40+x][(1-j)*40+y]=me->getBinContent(x+1,y+1)/eCount;
	       }
	    }
	 }
      }
   }

   //The global-summary
   //ReportSummary Map 
   //  ES+F  ES-F
   //  ES+R  ES-R
   float nGlobalErrors = 0;
   float nValidChannels = 0;
   float nErrorKinds = 0;

   if(UseDCC) nErrorKinds+=1;
   if(UseKChip) nErrorKinds+=1;
   if(UseDigi) nErrorKinds+=1;

   me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryMap");
   if(me){
      float xval, ErrorCount;
      for(int x=0; x<80; x++){
	 for(int y=0; y<80; y++){
	    //All three kind error has same weight
	    //Fill good efficiency into reportSummaryMap
	    xval = 0.;
	    ErrorCount = 0.;
	    if(Digi[x][y]>=1000&&UseDigi) ErrorCount+=1.;
	    if(DCC[x][y]!=3.&&UseDCC) ErrorCount+=1.;
	    if(KChip[x][y]!=3&&UseKChip) ErrorCount+=1.;

	    if( nErrorKinds != 0 ) xval = 1. - ErrorCount/nErrorKinds;

	    if(DCC[x][y]==0.&&UseDCC) xval = -1;	//-1 for non-module points
	    if(KChip[x][y]==0.&&UseKChip) xval = -1;	//-1 for non-module points

	    //Count Valid Channels and Error Channels
	    if( xval != -1 ){
	       nValidChannels+=1.;
	       nGlobalErrors = nGlobalErrors + ErrorCount/nErrorKinds;
	    }
	    me->setBinContent(x+1, y+1, xval);
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


