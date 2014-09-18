#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "DQM/EcalPreshowerMonitorClient/interface/ESSummaryClient.h"

using namespace std;

ESSummaryClient::ESSummaryClient(const edm::ParameterSet& ps) :
  ESClient(ps)
{
}

ESSummaryClient::~ESSummaryClient() {
}

void ESSummaryClient::book(DQMStore::IBooker& _ibooker) {

   if ( debug_ ) cout << "ESSummaryClient: setup" << endl;

   char histo[200];

   MonitorElement* me;

   _ibooker.setCurrentFolder( prefixME_ + "/EventInfo" );

   sprintf(histo, "reportSummary");
   me = _ibooker.bookFloat(histo);
   me->Fill(-1.0);      

   _ibooker.setCurrentFolder( prefixME_ + "/EventInfo/reportSummaryContents" );

   for (int i=0 ; i<2; ++i){
      for (int j=0 ; j<2; ++j){
	 int iz = (i==0)? 1:-1;
	 sprintf(histo, "EcalPreshower Z %d P %d", iz, j+1);
         me = _ibooker.bookFloat(histo);
         me->Fill(-1.0);      
      }
   }

   _ibooker.setCurrentFolder( prefixME_ + "/EventInfo" );

   sprintf(histo, "reportSummaryMap");
   me = _ibooker.book2D(histo, histo, 80, 0.5, 80.5, 80, 0.5, 80.5);
   me->setAxisTitle("Si X", 1);
   me->setAxisTitle("Si Y", 2);

   for ( int i = 0; i < 80; i++ ) {
      for ( int j = 0; j < 80; j++ ) {
	 me->setBinContent( i+1, j+1, -1. );
      }
   }

}

void ESSummaryClient::endLumiAnalyze(DQMStore::IGetter& _igetter) {

   char histo[200];

   float nDI_FedErr[80][80];
   float DCC[80][80];
   float eCount;

   MonitorElement* me;

   for (int i=0; i<80; ++i) 
     for (int j=0; j<80; ++j) {
	 nDI_FedErr[i][j] = -1;
	 DCC[i][j]=0;
      }

   for (int i=0; i<2; ++i) {
      for (int j=0; j<2; ++j) {

	 int iz = (i==0)? 1:-1;

	 sprintf(histo, "ES Integrity Errors Z %d P %d", iz, j+1);
	 me = _igetter.get(prefixME_ + "/ESIntegrityTask/" + histo);
	 if (me) 
	   for (int x=0; x<40; ++x) 
	     for (int y=0; y<40; ++y) 
	       nDI_FedErr[i*40+x][(1-j)*40+y] = me->getBinContent(x+1, y+1);
	 
	 sprintf(histo, "ES Integrity Summary 1 Z %d P %d", iz, j+1);
	 me = _igetter.get(prefixME_ + "/ESIntegrityClient/" + histo);
	 if (me)
	   for (int x=0; x<40; ++x)
	     for (int y=0; y<40; ++y) 
	       DCC[i*40+x][(1-j)*40+y] = me->getBinContent(x+1, y+1);
	 
	 sprintf(histo, "ES RecHit 2D Occupancy Z %d P %d", iz, j+1);
	 me = _igetter.get(prefixME_ + "/ESOccupancyTask/" + histo);
	 if (me)
	   eCount = me->getBinContent(40,40);
	 else 
	   eCount = 1.;
	 
      }
   }
   
   //The global-summary
   //ReportSummary Map 
   //  ES+F  ES-F
   //  ES+R  ES-R
   float nValidChannels=0; 
   float nGlobalErrors=0;
   float nValidChannelsES[2][2]={}; 
   float nGlobalErrorsES[2][2]={};

   me = _igetter.get(prefixME_ + "/EventInfo/reportSummaryMap");
   if (me) {
      for (int x=0; x<80; ++x) {
	if (eCount < 1) break; //Fill reportSummaryMap after have 1 event
	for (int y=0; y<80; ++y) {

	  int z = (x<40) ? 0:1;
	  int p = (y>=40) ? 0:1;
	  
	  if (DCC[x][y]==0.) {
	    me->setBinContent(x+1, y+1, -1.);	  
	  } else {
	    if (nDI_FedErr[x][y] >= 0) {
	      me->setBinContent(x+1, y+1, 1-(nDI_FedErr[x][y]/eCount));
	      
	      nValidChannels++;
	      nGlobalErrors += nDI_FedErr[x][y]/eCount;
	      
	      nValidChannelsES[z][p]++;
	      nGlobalErrorsES[z][p] += nDI_FedErr[x][y]/eCount;
	    }
	    else {
	      me->setBinContent(x+1, y+1, -1.);
	    }
	  }
	  
	}
      }
   }

   // Past version (<= CMSSW_7_2_0_pre4) of this module was filling reportSummaryContents and reportSummary with more information when the analyze function was called.
   // Since CMSSW_7_1_X, Client modules are not called per event any more.

   MonitorElement* me_report = 0;
   sprintf(histo, "ES Good Channel Fraction");
   me = _igetter.get(prefixME_+"/ESIntegrityTask/"+histo);
   if (!me) return;
   for (int i=0; i<2; ++i) {
     for (int j=0; j<2; ++j) {
       int iz = (i==0)? 1:-1;
       sprintf(histo, "EcalPreshower Z %d P %d", iz, j+1);
       me_report = _igetter.get(prefixME_+"/EventInfo/reportSummaryContents/" + histo);
       if (me_report) {
	 me_report->Fill(me->getBinContent(i+1, j+1));  
       }
     }
   }
   me_report = _igetter.get(prefixME_ + "/EventInfo/reportSummary");
   if ( me_report ) me_report->Fill(me->getBinContent(3,3));
}
