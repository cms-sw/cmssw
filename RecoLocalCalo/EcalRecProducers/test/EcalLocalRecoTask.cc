/*
 * \file EcalLocalReco.cc
 *
 * $Id: EcalLocalRecoTask.cc,v 1.12 2012/01/31 19:22:06 wmtan Exp $
 *
*/

#include <RecoLocalCalo/EcalRecProducers/test/EcalLocalRecoTask.h>
#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDetId/interface/EEDetId.h>
#include <DataFormats/EcalDetId/interface/ESDetId.h>

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalMGPAGainRatio.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"

#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

EcalLocalRecoTask::EcalLocalRecoTask(const edm::ParameterSet& ps)
{
  
  // DQM ROOT output
  outputFile_ = ps.getUntrackedParameter<std::string>("outputFile", "");

  recHitProducer_   = ps.getParameter<std::string>("recHitProducer");
  ESrecHitProducer_   = ps.getParameter<std::string>("ESrecHitProducer");
  EBrechitCollection_        = ps.getParameter<std::string>("EBrechitCollection");
  EErechitCollection_        = ps.getParameter<std::string>("EErechitCollection");
  ESrechitCollection_        = ps.getParameter<std::string>("ESrechitCollection");

  uncalibrecHitProducer_   = ps.getParameter<std::string>("uncalibrecHitProducer");
  EBuncalibrechitCollection_        = ps.getParameter<std::string>("EBuncalibrechitCollection");
  EEuncalibrechitCollection_        = ps.getParameter<std::string>("EEuncalibrechitCollection");

  digiProducer_   = ps.getParameter<std::string>("digiProducer");
  EBdigiCollection_        = ps.getParameter<std::string>("EBdigiCollection");
  EEdigiCollection_        = ps.getParameter<std::string>("EEdigiCollection");
  ESdigiCollection_        = ps.getParameter<std::string>("ESdigiCollection");

  
  if ( outputFile_.size() != 0 ) {
    edm::LogInfo("EcalLocalRecoTaskInfo") << "histograms will be saved to '" << outputFile_.c_str() << "'";
  } else {
    edm::LogInfo("EcalLocalRecoTaskInfo") << "histograms will NOT be saved";
  }
  
  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);
 
  if ( verbose_ ) {
    edm::LogInfo("EcalLocalRecoTaskInfo") << "verbose switch is ON"; 
  } else {
    edm::LogInfo("EcalLocalRecoTaskInfo") << "verbose switch is OFF";
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


  meEBUncalibRecHitMaxSampleRatio_ = 0;
  meEBUncalibRecHitPedestal_ = 0;
  meEBUncalibRecHitOccupancy_ = 0;
  meEBRecHitSimHitRatio_ = 0;

  Char_t histo[70];
 
  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalLocalRecoTask");
  
    sprintf (histo, "EcalLocalRecoTask Barrel occupancy" ) ;
    meEBUncalibRecHitOccupancy_ = dbe_->book2D(histo, histo, 360, 0., 360., 170, -85., 85.);

    sprintf (histo, "EcalLocalRecoTask Barrel Reco pedestals" ) ;
    meEBUncalibRecHitPedestal_ = dbe_->book1D(histo, histo, 1000, 0., 1000.);

    sprintf (histo, "EcalLocalRecoTask RecHit Max Sample Ratio") ;
    meEBUncalibRecHitMaxSampleRatio_ = dbe_->book1D(histo, histo, 200, 0., 2.);

    sprintf (histo, "EcalLocalRecoTask RecHit SimHit Ratio") ;
    meEBRecHitSimHitRatio_ = dbe_->book1D(histo, histo, 200, 0., 2.);
    
  }
 
}

EcalLocalRecoTask::~EcalLocalRecoTask(){
 
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);

}

void EcalLocalRecoTask::beginJob(){

}

void EcalLocalRecoTask::endJob(){

}

void EcalLocalRecoTask::analyze(const edm::Event& e, const edm::EventSetup& c)
{

  edm::LogInfo("EcalLocalRecoTaskInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();
  
   edm::Handle< EBDigiCollection > pEBDigis;
   edm::Handle< EEDigiCollection > pEEDigis;
   edm::Handle< ESDigiCollection > pESDigis;

   try {
     //     e.getByLabel( digiProducer_, EBdigiCollection_, pEBDigis);
     e.getByLabel( digiProducer_, pEBDigis);
   } catch ( std::exception& ex ) {
     edm::LogError("EcalLocalRecoTaskError") << "Error! can't get the product " << EBdigiCollection_.c_str() << std::endl;
   }

   try {
     //     e.getByLabel( digiProducer_, EEdigiCollection_, pEEDigis);
     e.getByLabel( digiProducer_, pEEDigis);
   } catch ( std::exception& ex ) {
     edm::LogError("EcalLocalRecoTaskError") << "Error! can't get the product " << EEdigiCollection_.c_str() << std::endl;
   }

   try {
     //     e.getByLabel( digiProducer_, ESdigiCollection_, pESDigis);
     e.getByLabel( digiProducer_, pESDigis);
   } catch ( std::exception& ex ) {
     edm::LogError("EcalLocalRecoTaskError") << "Error! can't get the product " << ESdigiCollection_.c_str() << std::endl;
   }


   edm::Handle< EBUncalibratedRecHitCollection > pEBUncalibRecHit;
   edm::Handle< EEUncalibratedRecHitCollection > pEEUncalibRecHit;

   try {
     e.getByLabel( uncalibrecHitProducer_, EBuncalibrechitCollection_, pEBUncalibRecHit);
   } catch ( std::exception& ex ) {
     edm::LogError("EcalLocalRecoTaskError") << "Error! can't get the product " << EBuncalibrechitCollection_.c_str() << std::endl;
   }
   try {
     e.getByLabel( uncalibrecHitProducer_, EEuncalibrechitCollection_, pEEUncalibRecHit);
   } catch ( std::exception& ex ) {
     edm::LogError("EcalLocalRecoTaskError") << "Error! can't get the product " << EEuncalibrechitCollection_.c_str() << std::endl;
   }


  edm::Handle<EBRecHitCollection> pEBRecHit;
  edm::Handle<EERecHitCollection> pEERecHit;
  edm::Handle<ESRecHitCollection> pESRecHit;

   try {
     e.getByLabel( recHitProducer_, EBrechitCollection_, pEBRecHit);
   } catch ( std::exception& ex ) {
     edm::LogError("EcalLocalRecoTaskError") << "Error! can't get the product " << EBrechitCollection_.c_str() << std::endl;
   }
   try {
     e.getByLabel( recHitProducer_, EErechitCollection_, pEERecHit);
   } catch ( std::exception& ex ) {
     edm::LogError("EcalLocalRecoTaskError") << "Error! can't get the product " << EErechitCollection_.c_str() << std::endl;
   }
   try {
     e.getByLabel( ESrecHitProducer_, ESrechitCollection_, pESRecHit);
   } catch ( std::exception& ex ) {
     edm::LogError("EcalLocalRecoTaskError") << "Error! can't get the product " << ESrechitCollection_.c_str() << std::endl;
   }


  edm::Handle< CrossingFrame<PCaloHit> > crossingFrame;
  const std::string barrelHitsName ("EcalHitsEB") ;
  try 
    {
      e.getByLabel("mix",barrelHitsName,crossingFrame);
    } catch ( std::exception& ex ) {
      edm::LogError("EcalLocalRecoTaskError") << "Error! can't get the crossingFrame" << std::endl;
    }

  edm::ESHandle<EcalPedestals> pPeds;
  try 
    {
      c.get<EcalPedestalsRcd>().get(pPeds);
    } catch ( std::exception& ex ) {
      edm::LogError("EcalLocalRecoTaskError") << "Error! can't get the Ecal pedestals" << std::endl;
    }

//   edm::ESHandle<EcalADCToGeVConstant> pAgc;
//   try 
//     {
//       c.get<EcalADCToGeVConstantRcd>().get(pAgc);
//     }
//   catch ( std::exception& ex ) 
//     {
//       edm::LogError("EcalLocalRecoTaskError") << "Error! can't get the Ecal ADCToGeV Constant" << std::endl;
//     } 


  std::auto_ptr<MixCollection<PCaloHit> > 
    barrelHits (new MixCollection<PCaloHit>(crossingFrame.product ())) ;

  MapType EBSimMap;
  
  for (MixCollection<PCaloHit>::MixItr hitItr = barrelHits->begin () ;
       hitItr != barrelHits->end () ;
       ++hitItr) {
    
    EBDetId EBid = EBDetId(hitItr->id()) ;
    
    LogDebug("EcalLocalRecoTaskDebug") 
      <<" CaloHit " << hitItr->getName() << " DetID = "<<hitItr->id()<< "\n"	
      << "Energy = " << hitItr->energy() << " Time = " << hitItr->time() << "\n"
      << "EBDetId = " << EBid.ieta() << " " << EBid.iphi();
    
    uint32_t crystid = EBid.rawId();
    EBSimMap[crystid] += hitItr->energy();
    
  }

  const EBDigiCollection * EBDigi = pEBDigis.product () ;
  const EBUncalibratedRecHitCollection * EBUncalibRecHit = pEBUncalibRecHit.product () ;
  const EBRecHitCollection * EBRecHit = pEBRecHit.product () ;

  //const EEDigiCollection * EEDigi = pEEDigis.product () ;
  //const EEUncalibratedRecHitCollection * EEUncalibRecHit = pEEUncalibRecHit.product () ;
  //const EERecHitCollection * EERecHit = pEERecHit.product () ;

  //const ESDigiCollection * ESDigi = pESDigis.product () ;
  //const ESRecHitCollection * ESRecHit = pESRecHit.product () ;


  // loop over uncalibRecHit
  for (EcalUncalibratedRecHitCollection::const_iterator uncalibRecHit = EBUncalibRecHit->begin () ;
       uncalibRecHit != EBUncalibRecHit->end () ;
       ++uncalibRecHit)
    {
      EBDetId EBid = EBDetId(uncalibRecHit->id());
      if (meEBUncalibRecHitOccupancy_) meEBUncalibRecHitOccupancy_->Fill( EBid.iphi(), EBid.ieta() );
      if (meEBUncalibRecHitPedestal_) meEBUncalibRecHitPedestal_->Fill(uncalibRecHit->pedestal());

      // Find corresponding recHit
      EcalRecHitCollection::const_iterator myRecHit = EBRecHit->find(EBid);
      // Find corresponding digi
      EBDigiCollection::const_iterator myDigi = EBDigi->find(EBid);

      double eMax = 0. ;

      if (myDigi != EBDigi->end())
	{
	  for (unsigned int sample = 0 ; sample < myDigi->size () ; ++sample)
	    {
	      double analogSample=EcalMGPASample((*myDigi)[sample]).adc() ;
	      if ( eMax < analogSample )
		{
		  eMax = analogSample;
		}
	    }
	}
      else
	continue;
      
      const EcalPedestals* myped=pPeds.product();
      EcalPedestals::const_iterator it = myped->getMap().find( EBid );
      if( it!=myped->getMap().end() )
	{
	  if (eMax> (*it).mean_x1 + 5 * (*it).rms_x1 ) //only real signal RecHit
	    {
	      if (meEBUncalibRecHitMaxSampleRatio_) meEBUncalibRecHitMaxSampleRatio_->Fill( (uncalibRecHit->amplitude()+uncalibRecHit->pedestal()) /eMax);
	      edm::LogInfo("EcalLocalRecoTaskInfo") << " eMax = " << eMax << " Amplitude = " << uncalibRecHit->amplitude()+uncalibRecHit->pedestal();  
	    }
	  else
	    continue;
	}
      else
	continue;
      
      if (myRecHit != EBRecHit->end())
	{
	  if ( EBSimMap[EBid.rawId()] != 0. ) 
	    if (meEBRecHitSimHitRatio_) 
	      meEBRecHitSimHitRatio_->Fill(myRecHit->energy()/EBSimMap[EBid.rawId()]);
	}
      else
	continue;
    }
}

//define this as a plug-in
DEFINE_FWK_MODULE(EcalLocalRecoTask);
                                                                                                                                                             
