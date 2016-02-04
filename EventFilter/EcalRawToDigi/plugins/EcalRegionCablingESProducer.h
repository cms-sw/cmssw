#ifndef EventFilter_EcalRawToDigi_EcalRegionCablingESProducer_H
#define EventFilter_EcalRawToDigi_EcalRegionCablingESProducer_H
// -*- C++ -*-
//
// Package:    EcalRegionCablingESProducer
// Class:      EcalRegionCablingESProducer
// 
/**\class EcalRegionCablingESProducer EcalRegionCablingESProducer.h EventFilter/EcalRegionCablingESProducer/src/EcalRegionCablingESProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jean-Roch Vlimant
//         Created:  Sun Oct  7 00:37:06 CEST 2007
// $Id: EcalRegionCablingESProducer.h,v 1.1 2007/12/06 20:01:11 vlimant Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "EventFilter/EcalRawToDigi/interface/EcalRegionCablingRecord.h"
#include "EventFilter/EcalRawToDigi/interface/EcalRegionCabling.h"

class EcalRegionCablingESProducer : public edm::ESProducer {
public:
      EcalRegionCablingESProducer(const edm::ParameterSet&);
  ~EcalRegionCablingESProducer();

  typedef boost::shared_ptr<EcalRegionCabling> ReturnType;

  ReturnType produce(const EcalRegionCablingRecord&);
private:
  edm::ParameterSet conf_;
};
#endif
