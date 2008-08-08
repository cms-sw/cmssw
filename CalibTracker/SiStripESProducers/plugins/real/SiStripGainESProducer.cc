// -*- C++ -*-
//
// Package:    SiStripGainESProducer
// Class:      SiStripGainESProducer
// 
/**\class SiStripGainESProducer SiStripGainESProducer.h CalibTracker/SiStripESProducer/plugins/real/SiStripGainESProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Giacomo Bruno
//         Created:  Fri Apr 27 12:31:25 CEST 2007
// $Id: SiStripGainESProducer.cc,v 1.3 2008/08/08 07:59:01 giordano Exp $
//
//



// user include files

#include "FWCore/Framework/interface/ESHandle.h"
#include "CalibTracker/SiStripESProducers/plugins/real/SiStripGainESProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


//
// constructors and destructor
//
SiStripGainESProducer::SiStripGainESProducer(const edm::ParameterSet& iConfig){

  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);
  
  //now do what ever other initialization is needed
  automaticMode_ = iConfig.getParameter<bool>("AutomaticNormalization");
  norm_=iConfig.getParameter<double>("NormalizationFactor");
  printdebug_ = iConfig.getUntrackedParameter<bool>("printDebug", false);
  apvgain_ = iConfig.getParameter<std::string>("APVGain");

  if(!automaticMode_ && norm_<=0){
    edm::LogError("SiStripGainESProducer::SiStripGainESProducer() - ERROR: negative or zero Normalization factor provided. Assuming 1 for such factor") << std::endl;
    norm_=1.;
  }

}


SiStripGainESProducer::~SiStripGainESProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
std::auto_ptr<SiStripGain> SiStripGainESProducer::produce(const SiStripGainRcd & iRecord)
{
   using namespace edm::es;
   edm::ESHandle<SiStripApvGain> pDD;
   iRecord.getRecord<SiStripApvGainRcd>().get(apvgain_,pDD );
   
   double NFactor;

   if(automaticMode_ || printdebug_ ){

     std::vector<uint32_t> DetIds;
     pDD->getDetIds(DetIds);

     double SumOfGains=0;
     int NGains=0;

     for(std::vector<uint32_t>::const_iterator detit=DetIds.begin(); detit!=DetIds.end(); detit++){

       SiStripApvGain::Range detRange = pDD->getRange(*detit);

       int iComp=0;

       for(std::vector<float>::const_iterator apvit=detRange.first; apvit!=detRange.second; apvit++){

	 SumOfGains+=(*apvit);
	 NGains++;
	 if (printdebug_)
	   edm::LogInfo("SiStripGainESProducer::produce()")<< "detid/component: " << *detit <<"/"<<iComp<< "   gain factor " <<*apvit ;
	 iComp++;


       }

     }

     if(automaticMode_){
       if(SumOfGains>0 && NGains>0){
	 NFactor=SumOfGains/NGains;
       }
       else{
	 edm::LogError("SiStripGainESProducer::produce() - ERROR: empty set of gain values received. Cannot compute normalization factor. Assuming 1 for such factor") << std::endl;
	 NFactor=1.;
       }
     }


   }

   if(!automaticMode_){
     NFactor=norm_;
   }


   if (printdebug_)  edm::LogInfo("SiStripGainESProducer::produce()")<< "putting A SiStrip Gain object in eventSetup with normalization factor " << NFactor ;

   SiStripGain * gain = new SiStripGain( *(pDD.product()), NFactor);
   return std::auto_ptr<SiStripGain>(gain );

}

