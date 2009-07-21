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
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"


#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalPreshowerMonitorModule/interface/ESOccupancyTask.h"

#include "TStyle.h"
#include "TH2F.h"

using namespace cms;
using namespace edm;
using namespace std;

ESOccupancyTask::ESOccupancyTask(const edm::ParameterSet& ps)
{

   rechitlabel_    = ps.getParameter<InputTag>("RecHitLabel");
   digilabel_    	= ps.getParameter<InputTag>("DigiLabel");
   prefixME_	= ps.getUntrackedParameter<string>("prefixME", "EcalPreshower"); 

   dqmStore_	= Service<DQMStore>().operator->();
   eCount_ = 0;

   //Histogram init  
   for(int i = 0; i < 2; i++) {
      for(int j = 0; j < 2; j++) {
	 hRecOCC_[i][j]=0;
	 hRecNHit_[i][j]=0;
	 hEng_[i][j]=0;
	 hEvEng_[i][j]=0;
	 hDigiOCC_[i][j]=0;
	 hDigiNHit_[i][j]=0;
      }
   }

   for(int i = 0; i<2; i++){
      hE1E2_[i]=0;
   }

   dqmStore_->setCurrentFolder(prefixME_ + "/ESOccupancyTask");

   //Booking Histograms
   //Notice: Change ESRenderPlugin under DQM/RenderPlugins/src if you change this histogram name.
   char histo[200];
   for (int i=0 ; i<2; ++i) { 
      for (int j=0 ; j<2; ++j) {
	 int iz = (i==0)? 1:-1;
	 sprintf(histo, "ES RecHit 2D Occupancy Z %d P %d", iz, j+1);
	 hRecOCC_[i][j] = dqmStore_->book2D(histo, histo, 40, 0.5, 40.5, 40, 0.5, 40.5);
	 hRecOCC_[i][j]->setAxisTitle("Si X", 1);
	 hRecOCC_[i][j]->setAxisTitle("Si Y", 2);

	 //Bin 40,40 is used to save eumber of event for scaling.
	 sprintf(histo, "ES Digi 2D Occupancy Z %d P %d", iz, j+1);
	 hDigiOCC_[i][j] = dqmStore_->book2D(histo, histo, 40, 0.5, 40.5, 40, 0.5, 40.5);
	 hDigiOCC_[i][j]->setAxisTitle("Si X", 1);
	 hDigiOCC_[i][j]->setAxisTitle("Si Y", 2);

	 sprintf(histo, "ES RecHit 1D Occupancy Z %d P %d", iz, j+1);
	 hRecNHit_[i][j] = dqmStore_->book1D(histo, histo, 30,0,300);
	 hRecNHit_[i][j]->setAxisTitle("RecHit Occupancy", 1);
	 hRecNHit_[i][j]->setAxisTitle("Num of Events", 2);

	 sprintf(histo, "ES Digi 1D Occupancy Z %d P %d", iz, j+1);
	 hDigiNHit_[i][j] = dqmStore_->book1D(histo, histo, 30,0,300);
	 hDigiNHit_[i][j]->setAxisTitle("Digi Occupancy", 1);
	 hDigiNHit_[i][j]->setAxisTitle("Num of Events", 2);

	 sprintf(histo, "ES RecHit Energy Z %d P %d", iz, j+1);
	 hEng_[i][j] = dqmStore_->book1D(histo, histo, 50, 0, 0.0005);
	 hEng_[i][j]->setAxisTitle("RecHit Energy", 1);
	 hEng_[i][j]->setAxisTitle("Num of ReHits", 2);

	 sprintf(histo, "ES Event Energy Z %d P %d", iz, j+1);
	 hEvEng_[i][j] = dqmStore_->book1D(histo, histo, 50, 0, 0.1);
	 hEvEng_[i][j]->setAxisTitle("Event Energy", 1);
	 hEvEng_[i][j]->setAxisTitle("Num of Events", 2);
      }
   }

   hE1E2_[0] = dqmStore_->book2D("ES+ EP1 vs EP2", "ES+ EP1 vs EP2", 50, 0, 0.1, 50, 0, 0.1);
   hE1E2_[1] = dqmStore_->book2D("ES- EP1 vs EP2", "ES- EP1 vs EP2", 50, 0, 0.1, 50, 0, 0.1);

}


ESOccupancyTask::~ESOccupancyTask(){}

void ESOccupancyTask::beginJob(const edm::EventSetup & c)
{
}

void ESOccupancyTask::endJob() {

   cout<<"Reach EndJob of ESOccupancyTask"<<endl;

}

void ESOccupancyTask::analyze(const edm::Event& e, const edm::EventSetup& iSetup)
{

   runNum_ = e.id().run();
   eCount_++;

   Handle<ESRecHitCollection> ESRecHit;
   try {
      e.getByLabel(rechitlabel_, ESRecHit);
   } catch ( cms::Exception &e ) {
      LogDebug("") << "Error! can't get RecHit collection !" << std::endl;
   }

   Handle<ESDigiCollection> digis;
   try {
      e.getByLabel(digilabel_, digis);
   } catch ( cms::Exception &e ) {
      LogDebug("") << "Error! can't get digi collection !" << std::endl;
   }

   // RecHits
   int zside, plane, ix, iy, strip;
   int sum_RecHits[2][2], sum_DigiHits[2][2];
   float sum_Energy[2][2];

   for(int i = 0; i < 2; i++ ) {
      for( int j = 0; j < 2; j++) {
	 sum_RecHits[i][j]=0;
	 sum_DigiHits[i][j]=0;
	 sum_Energy[i][j]=0;
      }
   }

   for (ESRecHitCollection::const_iterator hitItr = ESRecHit->begin(); hitItr != ESRecHit->end(); ++hitItr) 
   {

      ESDetId id = ESDetId(hitItr->id());

      zside = id.zside();
      plane = id.plane();
      ix    = id.six();
      iy    = id.siy();
      strip = id.strip();

      int i = (zside==1)? 0:1;
      int j = plane-1;

      sum_RecHits[i][j]++;
      sum_Energy[i][j]+=hitItr->energy();
      hRecOCC_[i][j]->Fill(ix, iy);
      if(hitItr->energy() != 0) hEng_[i][j]->Fill(hitItr->energy());

   }

   //DigiHits
   for (ESDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr) 
   {

      ESDataFrame dataframe = (*digiItr);
      ESDetId id = dataframe.id();

      zside = id.zside();
      plane = id.plane();
      ix    = id.six();
      iy    = id.siy();
      strip = id.strip();

      int i = (zside==1)? 0:1;
      int j = plane-1;

      sum_DigiHits[i][j]++;
      hDigiOCC_[i][j]->Fill(ix, iy, dataframe.sample(1).adc());

   }

   //Fill histograms after a event

   for( int i = 0; i < 2; i++){
      for( int j = 0; j < 2; j++){
	 if(sum_RecHits[i][j] != 0) hRecNHit_[i][j]->Fill(sum_RecHits[i][j]);
	 if(sum_DigiHits[i][j] != 0) hDigiNHit_[i][j]->Fill(sum_DigiHits[i][j]);
	 if(sum_DigiHits[i][j] != 0) hEvEng_[i][j]->Fill(sum_Energy[i][j]);

	 //Save eCount_ for Scaling
	 hRecOCC_[i][j]->setBinContent(40,40,eCount_);
	 hDigiOCC_[i][j]->setBinContent(40,40,eCount_);
      }
   }

   hE1E2_[0]->Fill(sum_Energy[0][0],sum_Energy[0][1]);
   hE1E2_[1]->Fill(sum_Energy[1][0],sum_Energy[1][2]);

}

//define this as a plug-in
DEFINE_FWK_MODULE(ESOccupancyTask);
