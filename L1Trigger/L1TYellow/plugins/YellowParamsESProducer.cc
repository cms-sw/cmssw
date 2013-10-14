///
/// \class l1t::
///
/// Description: Produces configuration parameters for the fictitious Yellow trigger.
///
/// Implementation:
///    Demonstrates how to create an ESProducer to fill CondFormats class with configuration data.
///
/// \author: Michael Mulhearn - UC Davis
///

//
//  Configuration data needed for the emulator should be received in the same
//  way, regardless of whether it is fetched from a database or a local
//  configuration file.
//
//  This ES Producer class fills the CondFormats class (YellowParams) which is
//  needed by YellowProducer.  It fills YellowParams from settings in a config
//  file (L1Trigger/L1TYellow/python/l1tyellow_params_cfi.py), but
//  YellowProducer receives it exactly as it would from the database.
//




// system include files
#include <memory>
#include "boost/shared_ptr.hpp"
#include <iostream>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/ESProducts.h"

#include "CondFormats/L1TYellow/interface/YellowParams.h"
#include "CondFormats/DataRecord/interface/L1TYellowParamsRcd.h"

using namespace std;

//
// class declaration
//

namespace l1t {

class YellowParamsESProducer : public edm::ESProducer {
public:
  YellowParamsESProducer(const edm::ParameterSet&);
  ~YellowParamsESProducer();
  
  typedef boost::shared_ptr<YellowParams> ReturnType;
  
  ReturnType produce(const L1TYellowParamsRcd&);

private:
  YellowParams  m_params ;
  std::string m_label;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
YellowParamsESProducer::YellowParamsESProducer(const edm::ParameterSet& conf)
{

   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);
   //setWhatProduced(this, conf.getParameter<std::string>("label"));

   m_params.setFirmwareVersion(conf.getParameter<unsigned>("firmwareVersion"));
   m_params.setParamA(conf.getParameter<unsigned>("paramA"));
   m_params.setParamB(conf.getParameter<unsigned>("paramB"));
   m_params.setParamC(conf.getParameter<double>("paramC"));
}


YellowParamsESProducer::~YellowParamsESProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
YellowParamsESProducer::ReturnType
YellowParamsESProducer::produce(const L1TYellowParamsRcd& iRecord)
{
   using namespace edm::es;
   boost::shared_ptr<YellowParams> pYellowParams ;

   pYellowParams = boost::shared_ptr< YellowParams >(new YellowParams( m_params ));
   return pYellowParams;
}

}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(l1t::YellowParamsESProducer);
