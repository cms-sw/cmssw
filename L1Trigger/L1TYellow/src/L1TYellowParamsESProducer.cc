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
#include <iostream>
#include "boost/shared_ptr.hpp"

using namespace std;

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TYellow/interface/L1TYellowParams.h"
#include "CondFormats/DataRecord/interface/L1TYellowParamsRcd.h"

// forward declarations

class L1TYellowParamsESProducer : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  L1TYellowParamsESProducer(const edm::ParameterSet&);
  ~L1TYellowParamsESProducer();

  typedef boost::shared_ptr<L1TYellowParams> ReturnType;

  ReturnType produce(const L1TYellowParamsRcd&);

protected:
  virtual void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
			      const edm::IOVSyncValue &,edm::ValidityInterval &);

private:
  // ----------member data ---------------------------
  L1TYellowParams  m_params ;
  std::string label;
};

void L1TYellowParamsESProducer::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
						const edm::IOVSyncValue & iosv, 
						edm::ValidityInterval & oValidity){
  edm::ValidityInterval infinity(iosv.beginOfTime(), iosv.endOfTime());
  oValidity = infinity;
}


L1TYellowParamsESProducer::L1TYellowParamsESProducer(const edm::ParameterSet& conf)
{
  m_params.setFirmwareVersion(conf.getParameter<unsigned>("firmwareVersion"));
  m_params.setParamA(conf.getParameter<unsigned>("paramA"));
  m_params.setParamB(conf.getParameter<unsigned>("paramB"));
  m_params.setParamC(conf.getParameter<unsigned>("paramC"));
  setWhatProduced(this, conf.getParameter<std::string>("label"));
  cout << "L1TYellow Params ESProducer Constructor Called" << "\n";
}

L1TYellowParamsESProducer::~L1TYellowParamsESProducer()
{
}

L1TYellowParamsESProducer::ReturnType
L1TYellowParamsESProducer::produce(const L1TYellowParamsRcd& iRecord)
{
  cout << "L1TYellow Params ESProducer produce method Called" << "\n";
  using namespace edm::es;
  boost::shared_ptr<L1TYellowParams> pL1TYellowParams ;

  pL1TYellowParams = boost::shared_ptr< L1TYellowParams >(new L1TYellowParams( m_params ));
  return pL1TYellowParams;
}

#include "FWCore/Framework/interface/SourceFactory.h"

DEFINE_FWK_EVENTSETUP_SOURCE(L1TYellowParamsESProducer);



