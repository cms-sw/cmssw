// -*- C++ -*-
//
// Package:    L1Trigger/L1TYellow
// Class:      L1TYellowParamsESProducer
// 
/**\class L1TYellowParamsESProducer L1TYellowParamsESProducer.cc L1Trigger/L1TYellow/src/L1TYellowParamsESProducer.cc

 Description:  This is part of the fictitious Yellow trigger emulation for demonstration purposes.

 This uses the parameters set in a config file to fill the ConfData/L1TYellow/L1TYellowParams object.  
 Other modules can retreive this object in same manner whether it was filled from config file or 
 the conditions database.

 Implementation:
     [Notes on implementation]
*/
//

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TYellow/interface/L1TYellowParams.h"
#include "CondFormats/DataRecord/interface/L1TYellowParamsRcd.h"

// forward declarations

class L1TYellowParamsESProducer : public edm::ESProducer {
public:
  L1TYellowParamsESProducer(const edm::ParameterSet&);
  ~L1TYellowParamsESProducer();

  typedef boost::shared_ptr<L1TYellowParams> ReturnType;

  ReturnType produce(const L1TYellowParamsRcd&);
private:
  // ----------member data ---------------------------
  L1TYellowParams  m_params ;
};


L1TYellowParamsESProducer::L1TYellowParamsESProducer(const edm::ParameterSet& iConfig)
{
  setWhatProduced(this, "L1TYellowParamsESProducer");
}
L1TYellowParamsESProducer::~L1TYellowParamsESProducer()
{
}

L1TYellowParamsESProducer::ReturnType
L1TYellowParamsESProducer::produce(const L1TYellowParamsRcd& iRecord)
{
  using namespace edm::es;
  boost::shared_ptr<L1TYellowParams> pL1TYellowParams ;

  pL1TYellowParams = boost::shared_ptr< L1TYellowParams >(new L1TYellowParams( m_params ));
  return pL1TYellowParams;
}

DEFINE_FWK_EVENTSETUP_MODULE(L1TYellowParamsESProducer);


