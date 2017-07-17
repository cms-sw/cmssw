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
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/EcalPreshowerMonitorModule/interface/ESOccupancyTask.h"

#include "TStyle.h"
#include "TH2F.h"

using namespace cms;
using namespace edm;
using namespace std;

ESOccupancyTask::ESOccupancyTask(const edm::ParameterSet& ps) {

  rechittoken_ = consumes<ESRecHitCollection>(ps.getParameter<InputTag>("RecHitLabel"));
  prefixME_	= ps.getUntrackedParameter<string>("prefixME", "EcalPreshower"); 
  
  eCount_ = 0;
  
  //Histogram init  
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) {
      hRecOCC_[i][j] = 0;
      hRecNHit_[i][j] = 0;
      hEng_[i][j] = 0;
      hEvEng_[i][j] = 0;
      hEnDensity_[i][i] = 0;
      hGoodRecNHit_[i][j] = 0;

      hSelEng_[i][j] = 0;
      hSelOCC_[i][j] = 0;
      hSelEnDensity_[i][j] = 0;
    }
  
  for (int i = 0; i<2; ++i) 
    hE1E2_[i]=0;
}

void
ESOccupancyTask::bookHistograms(DQMStore::IBooker& iBooker, Run const&, EventSetup const&)
{
  iBooker.setCurrentFolder(prefixME_ + "/ESOccupancyTask");
  
  //Booking Histograms
  //Notice: Change ESRenderPlugin under DQM/RenderPlugins/src if you change this histogram name.
  char histo[200];
  for (int i=0 ; i<2; ++i) 
    for (int j=0 ; j<2; ++j) {
      int iz = (i==0)? 1:-1;
      sprintf(histo, "ES RecHit 2D Occupancy Z %d P %d", iz, j+1);
      hRecOCC_[i][j] = iBooker.book2D(histo, histo, 40, 0.5, 40.5, 40, 0.5, 40.5);
      hRecOCC_[i][j]->setAxisTitle("Si X", 1);
      hRecOCC_[i][j]->setAxisTitle("Si Y", 2);
      
      //Bin 40,40 is used to save eumber of event for scaling.
      sprintf(histo, "ES Energy Density Z %d P %d", iz, j+1);
      hEnDensity_[i][j] = iBooker.book2D(histo, histo, 40, 0.5, 40.5, 40, 0.5, 40.5);
      hEnDensity_[i][j]->setAxisTitle("Si X", 1);
      hEnDensity_[i][j]->setAxisTitle("Si Y", 2);
      
      sprintf(histo, "ES Num of RecHits Z %d P %d", iz, j+1);
      hRecNHit_[i][j] = iBooker.book1DD(histo, histo, 60, 0, 1920);
      hRecNHit_[i][j]->setAxisTitle("# of RecHits", 1);
      hRecNHit_[i][j]->setAxisTitle("Num of Events", 2);
      
      sprintf(histo, "ES Num of Good RecHits Z %d P %d", iz, j+1);
      hGoodRecNHit_[i][j] = iBooker.book1DD(histo, histo, 60, 0, 1920);
      hGoodRecNHit_[i][j]->setAxisTitle("# of good RecHits", 1);
      hGoodRecNHit_[i][j]->setAxisTitle("Num of Events", 2);
      
      sprintf(histo, "ES RecHit Energy Z %d P %d", iz, j+1);
      hEng_[i][j] = iBooker.book1DD(histo, histo, 50, 0, 0.001);
      hEng_[i][j]->setAxisTitle("RecHit Energy", 1);
      hEng_[i][j]->setAxisTitle("Num of ReHits", 2);
      
      sprintf(histo, "ES Event Energy Z %d P %d", iz, j+1);
      hEvEng_[i][j] = iBooker.book1DD(histo, histo, 50, 0, 0.1);
      hEvEng_[i][j]->setAxisTitle("Event Energy", 1);
      hEvEng_[i][j]->setAxisTitle("Num of Events", 2);

      // histograms with selected hits
      sprintf(histo, "ES RecHit Energy with selected hits Z %d P %d", iz, j+1);
      hSelEng_[i][j] = iBooker.book1DD(histo, histo, 50, 0, 0.001);
      hSelEng_[i][j]->setAxisTitle("RecHit Energy", 1);
      hSelEng_[i][j]->setAxisTitle("Num of ReHits", 2);

      sprintf(histo, "ES Occupancy with selected hits Z %d P %d", iz, j+1);
      hSelOCC_[i][j] = iBooker.book2D(histo, histo, 40, 0.5, 40.5, 40, 0.5, 40.5);
      hSelOCC_[i][j]->setAxisTitle("Si X", 1);
      hSelOCC_[i][j]->setAxisTitle("Si Y", 2);

      sprintf(histo, "ES Energy Density with selected hits Z %d P %d", iz, j+1);
      hSelEnDensity_[i][j] = iBooker.book2D(histo, histo, 40, 0.5, 40.5, 40, 0.5, 40.5);
      hSelEnDensity_[i][j]->setAxisTitle("Si X", 1);
      hSelEnDensity_[i][j]->setAxisTitle("Si Y", 2);
    }

   hE1E2_[0] = iBooker.book2D("ES+ EP1 vs EP2", "ES+ EP1 vs EP2", 50, 0, 0.1, 50, 0, 0.1);
   hE1E2_[1] = iBooker.book2D("ES- EP1 vs EP2", "ES- EP1 vs EP2", 50, 0, 0.1, 50, 0, 0.1);
}

void ESOccupancyTask::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {

   runNum_ = e.id().run();
   eCount_++;

   // RecHits
   int zside, plane, ix, iy;
   int sum_RecHits[2][2], sum_GoodRecHits[2][2];
   float sum_Energy[2][2];

   for (int i = 0; i < 2; ++i) 
     for (int j = 0; j < 2; ++j) {
       sum_RecHits[i][j] = 0;
       sum_GoodRecHits[i][j] = 0;
       sum_Energy[i][j] = 0;
     }
   
   Handle<ESRecHitCollection> ESRecHit;
   if ( e.getByToken(rechittoken_, ESRecHit) ) {
     
     for (ESRecHitCollection::const_iterator hitItr = ESRecHit->begin(); hitItr != ESRecHit->end(); ++hitItr) {
       
       ESDetId id = ESDetId(hitItr->id());
       
       zside = id.zside();
       plane = id.plane();
       ix    = id.six();
       iy    = id.siy();
 
       int i = (zside==1)? 0:1;
       int j = plane-1;
       
       sum_RecHits[i][j]++;
       sum_Energy[i][j] += hitItr->energy();
       hRecOCC_[i][j]->Fill(ix, iy);
       if (hitItr->energy() != 0) {
	 hEng_[i][j]->Fill(hitItr->energy());
	 hEnDensity_[i][j]->Fill(ix, iy, hitItr->energy());

	 if (hitItr->recoFlag()==14 || hitItr->recoFlag()==1 || (hitItr->recoFlag()<=10 && hitItr->recoFlag()>=5)) continue;
	 sum_GoodRecHits[i][j]++;
	 hSelEng_[i][j]->Fill(hitItr->energy());
	 hSelEnDensity_[i][j]->Fill(ix, iy, hitItr->energy());
	 hSelOCC_[i][j]->Fill(ix, iy);
       }
       
     }
   } else {
     LogWarning("ESOccupancyTask") << "RecHitCollection not available";
   }

   //Fill histograms after a event
   for (int i = 0; i < 2; ++i) 
     for (int j = 0; j < 2; ++j) {

       hRecNHit_[i][j]->Fill(sum_RecHits[i][j]);
       hGoodRecNHit_[i][j]->Fill(sum_GoodRecHits[i][j]);
       hEvEng_[i][j]->Fill(sum_Energy[i][j]);
       
       //Save eCount_ for Scaling
       hRecOCC_[i][j]->setBinContent(40,40,eCount_);
       hEnDensity_[i][j]->setBinContent(40,40,eCount_);

       hSelOCC_[i][j]->setBinContent(40,40,eCount_);
       hSelEnDensity_[i][j]->setBinContent(40,40,eCount_);
     }

   hE1E2_[0]->Fill(sum_Energy[0][0], sum_Energy[0][1]);
   hE1E2_[1]->Fill(sum_Energy[1][0], sum_Energy[1][1]);

}

//define this as a plug-in
DEFINE_FWK_MODULE(ESOccupancyTask);
