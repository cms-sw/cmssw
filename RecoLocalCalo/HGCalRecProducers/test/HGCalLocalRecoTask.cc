/*
 * \file HGCalLocalRecoTask.cc
 *
 *
 *
*/

#include <RecoLocalCalo/HGCalRecProducers/test/HGCalLocalRecoTask.h>
#include <DataFormats/ForwardDetId/interface/HGCalDetId.h>
#include <DataFormats/ForwardDetId/interface/HGCEEDetId.h>
#include <DataFormats/ForwardDetId/interface/HGCHEDetId.h>

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

HGCalLocalRecoTask::HGCalLocalRecoTask(const edm::ParameterSet& ps)
{
  
  // DQM ROOT output
  outputFile_ = ps.getUntrackedParameter<std::string>("outputFile", "");

  recHitProducer_   = ps.getParameter<std::string>("recHitProducer");
  HGCEErechitCollection_        = ps.getParameter<std::string>("HGCEErechitCollection");
  HGCHEFrechitCollection_       = ps.getParameter<std::string>("HGCHEFrechitCollection");
  HGCHEBrechitCollection_       = ps.getParameter<std::string>("HGCHEBrechitCollection");

  uncalibrecHitProducer_   = ps.getParameter<std::string>("uncalibrecHitProducer");
  HGCEEuncalibrechitCollection_        = ps.getParameter<std::string>("HGCEEuncalibrechitCollection");
  HGCHEFuncalibrechitCollection_       = ps.getParameter<std::string>("HGCHEFuncalibrechitCollection");
  HGCHEBuncalibrechitCollection_       = ps.getParameter<std::string>("HGCHEBuncalibrechitCollection");

  digiProducer_   = ps.getParameter<std::string>("digiProducer");
  HGCEEdigiCollection_        = ps.getParameter<std::string>("HGCEEdigiCollection");
  HGCHEFdigiCollection_       = ps.getParameter<std::string>("HGCHEFdigiCollection");
  HGCHEBdigiCollection_       = ps.getParameter<std::string>("HGCHEBdigiCollection");

  
  if ( outputFile_.size() != 0 ) {
    edm::LogInfo("HGCalLocalRecoTaskInfo") << "histograms will be saved to '" << outputFile_.c_str() << "'";
  } else {
    edm::LogInfo("HGCalLocalRecoTaskInfo") << "histograms will NOT be saved";
  }
  
  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);
 
  if ( verbose_ ) {
    edm::LogInfo("HGCalLocalRecoTaskInfo") << "verbose switch is ON"; 
  } else {
    edm::LogInfo("HGCalLocalRecoTaskInfo") << "verbose switch is OFF";
  }
                                                                                                                                          
  dbe_ = 0;
                                                                                                                                          
  // get hold of back-end interface
  dbe_ = edm::Service<DQMStore>().operator->();
                                                                                                                                          
  if ( dbe_ ) {
    if ( verbose_ ) {
      dbe_->setVerbose(1);
    } else {
      dbe_->setVerbose(0);
    }
  }
                                                                                                                                          
  if ( dbe_ ) {
    if ( verbose_ ) dbe_->showDirStructure();
  }


  meHGCEEUncalibRecHitOccupancy_ = 0;
  meHGCHEFUncalibRecHitOccupancy_ = 0;
  meHGCHEBUncalibRecHitOccupancy_ = 0;

  Char_t histo[70];
 
  if ( dbe_ ) {
    dbe_->setCurrentFolder("HGCalLocalRecoTask");
  
    sprintf (histo, "HGCalLocalRecoTask HGCEE occupancy" ) ;
    meHGCEEUncalibRecHitOccupancy_ = dbe_->book2D(histo, histo, 360, 0., 360., 170, -85., 85.);

    sprintf (histo, "HGCalLocalRecoTask HGCHEF occupancy" ) ;
    meHGCHEFUncalibRecHitOccupancy_ = dbe_->book2D(histo, histo, 360, 0., 360., 170, -85., 85.);

    sprintf (histo, "HGCalLocalRecoTask HGCHEB occupancy" ) ;
    meHGCHEBUncalibRecHitOccupancy_ = dbe_->book2D(histo, histo, 360, 0., 360., 170, -85., 85.);
    
  }
 
}

HGCalLocalRecoTask::~HGCalLocalRecoTask(){
 
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);

}

void HGCalLocalRecoTask::beginJob(){

}

void HGCalLocalRecoTask::endJob(){

}

void HGCalLocalRecoTask::analyze(const edm::Event& e, const edm::EventSetup& c)
{

  edm::LogInfo("HGCalLocalRecoTaskInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();
  
   edm::Handle< HGCEEDigiCollection > pHGCEEDigis;
   edm::Handle< HGCHEDigiCollection > pHGCHEFDigis;
   edm::Handle< HGCHEDigiCollection > pHGCHEBDigis;

   try {
     //     e.getByLabel( digiProducer_, HGCEEdigiCollection_, pHGCEEDigis);
     e.getByLabel( digiProducer_, pHGCEEDigis);
   } catch ( std::exception& ex ) {
     edm::LogError("HGCalLocalRecoTaskError") << "Error! can't get the product " << HGCEEdigiCollection_.c_str() << std::endl;
   }

   try {
     e.getByLabel( digiProducer_, pHGCHEFDigis);
   } catch ( std::exception& ex ) {
     edm::LogError("HGCalLocalRecoTaskError") << "Error! can't get the product " << HGCHEFdigiCollection_.c_str() << std::endl;
   }

   try {
     e.getByLabel( digiProducer_, pHGCHEBDigis);
   } catch ( std::exception& ex ) {
     edm::LogError("HGCalLocalRecoTaskError") << "Error! can't get the product " << HGCHEBdigiCollection_.c_str() << std::endl;
   }


   edm::Handle< HGCeeUncalibratedRecHitCollection > pHGCEEUncalibRecHit;
   edm::Handle< HGChefUncalibratedRecHitCollection > pHGCHEFUncalibRecHit;
   edm::Handle< HGChebUncalibratedRecHitCollection > pHGCHEBUncalibRecHit;

   try {
     e.getByLabel( uncalibrecHitProducer_, HGCEEuncalibrechitCollection_, pHGCEEUncalibRecHit);
   } catch ( std::exception& ex ) {
     edm::LogError("HGCalLocalRecoTaskError") << "Error! can't get the product " << HGCEEuncalibrechitCollection_.c_str() << std::endl;
   }
   try {
     e.getByLabel( uncalibrecHitProducer_, HGCHEFuncalibrechitCollection_, pHGCHEFUncalibRecHit);
   } catch ( std::exception& ex ) {
     edm::LogError("HGCalLocalRecoTaskError") << "Error! can't get the product " << HGCHEFuncalibrechitCollection_.c_str() << std::endl;
   }
   try {
     e.getByLabel( uncalibrecHitProducer_, HGCHEBuncalibrechitCollection_, pHGCHEBUncalibRecHit);
   } catch ( std::exception& ex ) {
     edm::LogError("HGCalLocalRecoTaskError") << "Error! can't get the product " << HGCHEBuncalibrechitCollection_.c_str() << std::endl;
   }


  edm::Handle<HGCeeRecHitCollection> pHGCEERecHit;
  edm::Handle<HGChefRecHitCollection> pHGCHEFRecHit;
  edm::Handle<HGChebRecHitCollection> pHGCHEBRecHit;

   try {
     e.getByLabel( recHitProducer_, HGCEErechitCollection_, pHGCEERecHit);
   } catch ( std::exception& ex ) {
     edm::LogError("HGCalLocalRecoTaskError") << "Error! can't get the product " << HGCEErechitCollection_.c_str() << std::endl;
   }
   try {
     e.getByLabel( recHitProducer_, HGCHEFrechitCollection_, pHGCHEFRecHit);
   } catch ( std::exception& ex ) {
     edm::LogError("HGCalLocalRecoTaskError") << "Error! can't get the product " << HGCHEFrechitCollection_.c_str() << std::endl;
   }
   try {
     e.getByLabel( recHitProducer_, HGCHEBrechitCollection_, pHGCHEBRecHit);
   } catch ( std::exception& ex ) {
     edm::LogError("HGCalLocalRecoTaskError") << "Error! can't get the product " << HGCHEBrechitCollection_.c_str() << std::endl;
   }


  edm::Handle< CrossingFrame<PCaloHit> > crossingFrame;
  const std::string HGCEEHitsName ("HGCEEHits") ;
  try 
    {
      e.getByLabel("mix",HGCEEHitsName,crossingFrame);
    } catch ( std::exception& ex ) {
      edm::LogError("HGCalLocalRecoTaskError") << "Error! can't get the crossingFrame" << std::endl;
    }

  std::auto_ptr<MixCollection<PCaloHit> > 
    HGCEEHits (new MixCollection<PCaloHit>(crossingFrame.product ())) ;

  //  MapType HGCEESimMap;
  
  //  for (MixCollection<PCaloHit>::MixItr hitItr = HGCEEHits->begin () ;
  //       hitItr != HGCEEHits->end () ;
  //       ++hitItr) {
    
  //    HGCEEDetId HGCEEid = HGCEEDetId(hitItr->id()) ;
    
  //    LogDebug("HGCalLocalRecoTaskDebug") 
  //      <<" CaloHit " << hitItr->getName() << " DetID = "<<hitItr->id()<< "\n"	
  //      << "Energy = " << hitItr->energy() << " Time = " << hitItr->time() << "\n"
  //      << "HGCEEDetId = " << HGCEEid.ieta() << " " << HGCEEid.iphi();
    
  //    uint32_t cell_id = HGCEEid.rawId();
  //    HGCEESimMap[cell_id] += hitItr->energy();
    
  //  }

  const HGCEEDigiCollection * HGCEEDigi = pHGCEEDigis.product () ;
  const HGCeeUncalibratedRecHitCollection * HGCEEUncalibRecHit = pHGCEEUncalibRecHit.product () ;
  const HGCeeRecHitCollection * HGCEERecHit = pHGCEERecHit.product () ;

  // loop over uncalibRecHit
  for (HGCUncalibratedRecHitCollection::const_iterator uncalibRecHit = HGCEEUncalibRecHit->begin () ;
       uncalibRecHit != HGCEEUncalibRecHit->end () ;
       ++uncalibRecHit)
    {
      HGCEEDetId HGCEEid = HGCEEDetId(uncalibRecHit->id());
      if (meHGCEEUncalibRecHitOccupancy_) meHGCEEUncalibRecHitOccupancy_->Fill( HGCEEid.cell(), HGCEEid.sector() );

      // Find corresponding recHit
      HGCRecHitCollection::const_iterator myRecHit = HGCEERecHit->find(HGCEEid);
      // Find corresponding digi
      HGCEEDigiCollection::const_iterator myDigi = HGCEEDigi->find(HGCEEid);

      double eMax = 0. ;

      if (myDigi != HGCEEDigi->end())
	{
	  for (int sample = 0 ; sample < myDigi->size () ; ++sample)
	    {
	      double analogSample=HGCSample((*myDigi)[sample]).data() ;
	      if ( eMax < analogSample )
		{
		  eMax = analogSample;
		}
	    }
	}
      else
	continue;
      
      

    }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalLocalRecoTask);
                                                                                                                                                             
