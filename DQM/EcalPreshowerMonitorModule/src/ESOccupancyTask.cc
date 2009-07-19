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
   for(int i = 0; i<4; i++){
      hRecOCC_[i]=0;
      hRecNHit_[i]=0;
      hEng_[i]=0;
      hEvEng_[i]=0;
      hDigiOCC_[i]=0;
      hDigiNHit_[i]=0;
   }

   for(int i = 0; i<2; i++){
      hE1E2_[i]=0;
   }

   dqmStore_->setCurrentFolder(prefixME_ + "/ESOccupancyTask");

   //Booking Histograms
   //Notice: Change ESRenderPlugin under DQM/RenderPlugins/src if you change this histogram name.
   hRecNHit_[0] = dqmStore_->book1D("ES+ P1 RecHit 1D Occupancy", "ES+ P1 RecHit 1D Occupancy", 30,0,300); 
   hRecNHit_[1] = dqmStore_->book1D("ES+ P2 RecHit 1D Occupancy", "ES+ P2 RecHit 1D Occupancy", 30,0,300);
   hRecNHit_[2] = dqmStore_->book1D("ES- P1 RecHit 1D Occupancy", "ES- P1 RecHit 1D Occupancy", 30,0,300);
   hRecNHit_[3] = dqmStore_->book1D("ES- P2 RecHit 1D Occupancy", "ES- P2 RecHit 1D Occupancy", 30,0,300);

   //Bin 41,41 is used to save eumber of event for scaling.
   hRecOCC_[0] = dqmStore_->book2D("ES+ P1 RecHit 2D Occupancy", "ES+ P1 RecHit 2D Occupancy", 40, 0.5, 40.5, 40, 0.5, 40.5);
   hRecOCC_[1] = dqmStore_->book2D("ES+ P2 RecHit 2D Occupancy", "ES+ P2 RecHit 2D Occupancy", 40, 0.5, 40.5, 40, 0.5, 40.5);
   hRecOCC_[2] = dqmStore_->book2D("ES- P1 RecHit 2D Occupancy", "ES- P1 RecHit 2D Occupancy", 40, 0.5, 40.5, 40, 0.5, 40.5);
   hRecOCC_[3] = dqmStore_->book2D("ES- P2 RecHit 2D Occupancy", "ES- P2 RecHit 2D Occupancy", 40, 0.5, 40.5, 40, 0.5, 40.5);

   hEng_[0] = dqmStore_->book1D("ES+ P1 RecHit Energy", "ES+ P1 RecHit Energy", 50, 0, 0.0005);
   hEng_[1] = dqmStore_->book1D("ES+ P2 RecHit Energy", "ES+ P2 RecHit Energy", 50, 0, 0.0005);
   hEng_[2] = dqmStore_->book1D("ES- P1 RecHit Energy", "ES- P1 RecHit Energy", 50, 0, 0.0005);
   hEng_[3] = dqmStore_->book1D("ES- P2 RecHit Energy", "ES- P2 RecHit Energy", 50, 0, 0.0005);

   hEvEng_[0] = dqmStore_->book1D("ES+ P1 Event Energy", "ES+ P1 Event Energy", 50, 0, 0.1);
   hEvEng_[1] = dqmStore_->book1D("ES+ P2 Event Energy", "ES+ P2 Event Energy", 50, 0, 0.1);
   hEvEng_[2] = dqmStore_->book1D("ES- P1 Event Energy", "ES- P1 Event Energy", 50, 0, 0.1);
   hEvEng_[3] = dqmStore_->book1D("ES- P2 Event Energy", "ES- P2 Event Energy", 50, 0, 0.1);

   hE1E2_[0] = dqmStore_->book2D("ES+ EP1 vs EP2", "ES+ EP1 vs EP2", 50, 0, 0.1, 50, 0, 0.1);
   hE1E2_[1] = dqmStore_->book2D("ES- EP1 vs EP2", "ES- EP1 vs EP2", 50, 0, 0.1, 50, 0, 0.1);

   hDigiNHit_[0] = dqmStore_->book1D("ES+ P1 DigiHit 1D Occupancy", "ES+ P1 DigiHit 1D Occupancy", 30,0,300);
   hDigiNHit_[1] = dqmStore_->book1D("ES+ P2 DigiHit 1D Occupancy", "ES+ P2 DigiHit 1D Occupancy", 30,0,300);
   hDigiNHit_[2] = dqmStore_->book1D("ES- P1 DigiHit 1D Occupancy", "ES- P1 DigiHit 1D Occupancy", 30,0,300);
   hDigiNHit_[3] = dqmStore_->book1D("ES- P2 DigiHit 1D Occupancy", "ES- P2 DigiHit 1D Occupancy", 30,0,300);

   hDigiOCC_[0] = dqmStore_->book2D("ES+ P1 DigiHit 2D Occupancy", "ES+ P1 DigiHit 2D Occupancy", 40, 0.5, 40.5, 40, 0.5, 40.5);
   hDigiOCC_[1] = dqmStore_->book2D("ES+ P2 DigiHit 2D Occupancy", "ES+ P2 DigiHit 2D Occupancy", 40, 0.5, 40.5, 40, 0.5, 40.5);
   hDigiOCC_[2] = dqmStore_->book2D("ES- P1 DigiHit 2D Occupancy", "ES- P1 DigiHit 2D Occupancy", 40, 0.5, 40.5, 40, 0.5, 40.5);
   hDigiOCC_[3] = dqmStore_->book2D("ES- P2 DigiHit 2D Occupancy", "ES- P2 DigiHit 2D Occupancy", 40, 0.5, 40.5, 40, 0.5, 40.5);

}


ESOccupancyTask::~ESOccupancyTask(){}


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
   int sum_RecHits[4], sum_DigiHits[4];
   float sum_Energy[4];

   for(int i = 0; i < 4; i++ ){
      sum_RecHits[i]=0;
      sum_DigiHits[i]=0;
      sum_Energy[i]=0;
   }

   for (ESRecHitCollection::const_iterator hitItr = ESRecHit->begin(); hitItr != ESRecHit->end(); ++hitItr) {

      ESDetId id = ESDetId(hitItr->id());

      zside = id.zside();
      plane = id.plane();
      ix    = id.six();
      iy    = id.siy();
      strip = id.strip();

      if ( (zside == +1) && (plane == 1) ){
	 sum_RecHits[0]++;
	 sum_Energy[0]+=hitItr->energy();
	 hRecOCC_[0]->Fill(ix, iy);
	 hEng_[0]->Fill(hitItr->energy());
      }

      if( (zside == +1) && (plane == 2) ){
	 sum_RecHits[1]++;
	 sum_Energy[1]+=hitItr->energy();
	 hRecOCC_[1]->Fill(ix, iy);
	 hEng_[1]->Fill(hitItr->energy());
      }

      if( (zside == -1) && (plane == 1) ){
	 sum_RecHits[2]++;
	 sum_Energy[2]+=hitItr->energy();
	 hRecOCC_[2]->Fill(ix, iy);
	 hEng_[2]->Fill(hitItr->energy());
      }

      if( (zside == -1) && (plane == 2) ){
	 sum_RecHits[3]++;
	 sum_Energy[3]+=hitItr->energy();
	 hRecOCC_[3]->Fill(ix, iy);
	 hEng_[3]->Fill(hitItr->energy());
      }
   }

   //DigiHits
   for (ESDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr) {

      ESDataFrame dataframe = (*digiItr);
      ESDetId id = dataframe.id();

      zside = id.zside();
      plane = id.plane();
      ix    = id.six();
      iy    = id.siy();
      strip = id.strip();

      if ( (zside == +1) && (plane == 1) ){
	 sum_DigiHits[0]++;
	 hDigiOCC_[0]->Fill(ix, iy, dataframe.sample(1).adc());
      }

      if( (zside == +1) && (plane == 2) ){
	 sum_DigiHits[1]++;
	 hDigiOCC_[1]->Fill(ix, iy, dataframe.sample(1).adc());
      }

      if( (zside == -1) && (plane == 1) ){
	 sum_DigiHits[2]++;
	 hDigiOCC_[2]->Fill(ix, iy, dataframe.sample(1).adc());
      }

      if( (zside == -1) && (plane == 2) ){
	 sum_DigiHits[3]++;
	 hDigiOCC_[3]->Fill(ix, iy, dataframe.sample(1).adc());
      }
   }


   //Fill histograms after a event

   for(int i=0; i<4; i++){
      hRecNHit_[i]->Fill(sum_RecHits[i]);
      hDigiNHit_[i]->Fill(sum_DigiHits[i]);
      hEvEng_[i]->Fill(sum_Energy[i]);

      //Save eCount_ for Scaling
      hRecOCC_[i]->setBinContent(40,40,eCount_);
      hDigiOCC_[i]->setBinContent(40,40,eCount_);
   }

   hE1E2_[0]->Fill(sum_Energy[0],sum_Energy[1]);
   hE1E2_[1]->Fill(sum_Energy[2],sum_Energy[3]);


}


void ESOccupancyTask::beginJob(const edm::EventSetup & c)
{
}

void ESOccupancyTask::endJob() {

   cout<<"Reach EndJob of ESOccupancyTask"<<endl;
   //	float EvtWeight = eCount_;
   //	EvtWeight = 1./EvtWeight;
   //	for(int i=0;i<4;i++){
   //		hOCC_[i]->getTH2F()->Scale(EvtWeight);
   //	}

}

//define this as a plug-in
DEFINE_FWK_MODULE(ESOccupancyTask);
