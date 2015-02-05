///
/// \class L1TCaloConfigESProducer
///
/// Description: Produces configuration of Calo Trigger
///
/// Implementation:
///    Dummy producer for L1 calo upgrade runtime configuration
///


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"
#include <iostream>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "CondFormats/L1TObjects/interface/CaloConfig.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloConfigHelper.h"
#include "CondFormats/DataRecord/interface/L1TCaloConfigRcd.h"

using namespace std;

//
// class declaration
//

using namespace l1t;

class L1TCaloConfigESProducer : public edm::ESProducer {
public:
  L1TCaloConfigESProducer(const edm::ParameterSet&);
  ~L1TCaloConfigESProducer();

  typedef boost::shared_ptr<CaloConfig> ReturnType;

  ReturnType produce(const L1TCaloConfigRcd&);

private:
  CaloConfig  m_params;
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
L1TCaloConfigESProducer::L1TCaloConfigESProducer(const edm::ParameterSet& conf)
{
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);
  //setWhatProduced(this, conf.getParameter<std::string>("label"));

  std::string l1epoch = conf.getParameter<string>("l1Epoch");
  unsigned fwv = conf.getParameter<unsigned>("fwVersionLayer2");
  CaloConfigHelper h(m_params, fwv, l1epoch);
}


L1TCaloConfigESProducer::~L1TCaloConfigESProducer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
L1TCaloConfigESProducer::ReturnType
L1TCaloConfigESProducer::produce(const L1TCaloConfigRcd& iRecord)
{
   using namespace edm::es;
   boost::shared_ptr<CaloConfig> pCaloConfig ;
   pCaloConfig = boost::shared_ptr< CaloConfig >(new CaloConfig(m_params));
   return pCaloConfig;
}



//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TCaloConfigESProducer);
