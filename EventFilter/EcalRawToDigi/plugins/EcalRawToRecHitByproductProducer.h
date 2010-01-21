#ifndef EventFilter_EcalRawToDigi_EcalRawToRecHitByproductProducer_H
#define EventFilter_EcalRawToDigi_EcalRawToRecHitByproductProducer_H

/* \class EcalRawToRecHitByproductProducer
 *
 * Description:
 * EDProducer which puts in the event all the by-products of the EcalUnpackerWorker
 *
 * \author J-R Vlimant
*/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


//
// class decleration
//

class EcalRawToRecHitByproductProducer : public edm::EDProducer {
public:
  explicit EcalRawToRecHitByproductProducer(const edm::ParameterSet&);
  ~EcalRawToRecHitByproductProducer(){};
  
private:
  virtual void produce(edm::Event&, const edm::EventSetup&);

  std::string workerName_;
};

#endif
