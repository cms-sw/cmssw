// -*- C++ -*-
//
// Package:    L1Trigger/L1TYellowParamsESProducer
// Class:      L1TYellowParamsESProducer
// 
/**\class L1TYellowParamsESProducer L1TYellowParamsESProducer.h L1Trigger/L1TYellowParamsESProducer/plugins/L1TYellowParamsESProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/

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
   m_params.setParamC(conf.getParameter<unsigned>("paramC"));

   cout << "YellowParamsESProducer constructor called.\n";
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
   cout << "YellowParamsESProducer produce called!!!\n";


   using namespace edm::es;
   boost::shared_ptr<YellowParams> pYellowParams ;

   pYellowParams = boost::shared_ptr< YellowParams >(new YellowParams( m_params ));
   return pYellowParams;
}

}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(l1t::YellowParamsESProducer);
