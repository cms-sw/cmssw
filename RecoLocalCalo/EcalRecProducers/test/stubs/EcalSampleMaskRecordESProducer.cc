#include "boost/shared_ptr.hpp"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/EcalSampleMaskRcd.h"
#include "CondFormats/EcalObjects/interface/EcalSampleMask.h"

/**
   ESProducer to fill the EcalSampleMaskRecord
   starting from EcalChannelStatus information
   

   \author Stefano Argiro
   \version $Id: EcalSampleMaskRecordESProducer.cc,v 1.1 2012/05/10 08:23:23 argiro Exp $
   \date 9 May 2012
*/


class EcalSampleMaskRecordESProducer : public edm::ESProducer {
  
public:
  EcalSampleMaskRecordESProducer(const edm::ParameterSet& iConfig);
  
  typedef boost::shared_ptr<EcalSampleMask> ReturnType;
  
  ReturnType produce(const EcalSampleMaskRcd& iRecord);
  
  unsigned int maskeb_;
  unsigned int maskee_;

};


EcalSampleMaskRecordESProducer::EcalSampleMaskRecordESProducer(const edm::ParameterSet& iConfig){
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);

  maskeb_ = iConfig.getParameter<unsigned int>("maskeb");
  maskee_ = iConfig.getParameter<unsigned int>("maskee");

}



EcalSampleMaskRecordESProducer::ReturnType
EcalSampleMaskRecordESProducer::produce(const EcalSampleMaskRcd& iRecord){

  ReturnType returnRcd (new EcalSampleMask(maskeb_,maskee_));

  return returnRcd ;

}


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(EcalSampleMaskRecordESProducer);








// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "cd .. ; scram b"
// End:
