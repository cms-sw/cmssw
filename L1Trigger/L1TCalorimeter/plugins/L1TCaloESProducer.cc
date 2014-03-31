///
/// \class l1t::
///
/// Description: Produces configuration parameters for the fictitious Yellow trigger.
///
/// Implementation:
///    Dummy producer for L1 calo upgrade configuration parameters
///
/// \author: Jim Brooke, University of Bristol
///

//
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

#include "CondFormats/L1TObjects/interface/CaloParams.h"
#include "CondFormats/DataRecord/interface/L1TCaloParamsRcd.h"

using namespace std;

//
// class declaration
//

namespace l1t {

class L1TCaloParamsESProducer : public edm::ESProducer {
public:
  L1TCaloParamsESProducer(const edm::ParameterSet&);
  ~L1TCaloParamsESProducer();
  
  typedef boost::shared_ptr<CaloParams> ReturnType;
  
  ReturnType produce(const L1TCaloParamsRcd&);

private:
  CaloParams  m_params ;
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
L1TCaloParamsESProducer::L1TCaloParamsESProducer(const edm::ParameterSet& conf)
{

   //the following line is needed to tell the framework what
   // data is being produced
  setWhatProduced(this);
   //setWhatProduced(this, conf.getParameter<std::string>("label"));

  // tower scales
  m_params.setTowerLsbH(conf.getParameter<double>("towerLsbH"));
  m_params.setTowerLsbE(conf.getParameter<double>("towerLsbE"));
  m_params.setTowerLsbSum(conf.getParameter<double>("towerLsbSum"));
  m_params.setTowerNBitsH(conf.getParameter<int>("towerNBitsH"));
  m_params.setTowerNBitsE(conf.getParameter<int>("towerNBitsE"));
  m_params.setTowerNBitsSum(conf.getParameter<int>("towerNBitsSum"));
  m_params.setTowerNBitsRatio(conf.getParameter<int>("towerNBitsRatio"));
  m_params.setTowerCompression(conf.getParameter<bool>("towerCompression"));

  m_params.setJetSeedThreshold(conf.getParameter<double>("jetSeedThreshold"));

}


L1TCaloParamsESProducer::~L1TCaloParamsESProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
L1TCaloParamsESProducer::ReturnType
L1TCaloParamsESProducer::produce(const L1TCaloParamsRcd& iRecord)
{
   using namespace edm::es;
   boost::shared_ptr<CaloParams> pCaloParams ;

   pCaloParams = boost::shared_ptr< CaloParams >(new CaloParams(m_params));
   return pCaloParams;
}

}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(l1t::L1TCaloParamsESProducer);
