#ifndef CalibTracker_SiStripESProducers_SiStripGainESProducer_h
#define CalibTracker_SiStripESProducers_SiStripGainESProducer_h
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
// $Id: SiStripGainESProducer.h,v 1.1 2007/10/11 12:52:55 giordano Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"
#include "CondFormats/DataRecord/interface/SiStripApvGainRcd.h"



//
// class decleration
//

class SiStripGainESProducer : public edm::ESProducer {

   public:
      SiStripGainESProducer(const edm::ParameterSet &);
      ~SiStripGainESProducer();

  //      typedef edm::ESProducts<> ReturnType;

      std::auto_ptr<SiStripGain>  produce(const SiStripGainRcd &);

   private:
      // ----------member data ---------------------------

  double norm_;
  bool automaticMode_;
  bool  printdebug_;

};


#endif

//define this as a plug-in
//DEFINE_FWK_EVENTSETUP_MODULE(SiStripGainESProducer);
