///
/// \class L1TGlobalPrescalesVetosESProducer
///
/// Description: Produces L1T Trigger Menu Condition Format
///
/// Implementation:
///    Dummy producer for L1T uGT Trigger Menu
///


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#include "tmEventSetup/tmEventSetup.hh"
#include "tmEventSetup/esTriggerMenu.hh"
#include "tmEventSetup/esAlgorithm.hh"
#include "tmEventSetup/esCondition.hh"
#include "tmEventSetup/esObject.hh"
#include "tmEventSetup/esCut.hh"
#include "tmEventSetup/esScale.hh"
#include "tmGrammar/Algorithm.hh"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "L1Trigger/L1TGlobal/interface/PrescalesVetosHelper.h"

#include "CondFormats/L1TObjects/interface/L1TGlobalPrescalesVetos.h"
#include "CondFormats/DataRecord/interface/L1TGlobalPrescalesVetosRcd.h"

using namespace std;
using namespace edm;
using namespace l1t;
//
// class declaration
//

class L1TGlobalPrescalesVetosESProducer : public edm::ESProducer {
public:
  L1TGlobalPrescalesVetosESProducer(const edm::ParameterSet&);
  ~L1TGlobalPrescalesVetosESProducer();

  typedef boost::shared_ptr<L1TGlobalPrescalesVetos> ReturnType;

  ReturnType produce(const L1TGlobalPrescalesVetosRcd&);

private:

  PrescalesVetosHelper data_;
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
L1TGlobalPrescalesVetosESProducer::L1TGlobalPrescalesVetosESProducer(const edm::ParameterSet& conf) :
  data_(new L1TGlobalPrescalesVetos())
{
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);
  //setWhatProduced(this, conf.getParameter<std::string>("label"));


}


L1TGlobalPrescalesVetosESProducer::~L1TGlobalPrescalesVetosESProducer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
L1TGlobalPrescalesVetosESProducer::ReturnType
L1TGlobalPrescalesVetosESProducer::produce(const L1TGlobalPrescalesVetosRcd& iRecord)
{
  
  // configure the helper class parameters via its set funtions, e.g.:
  data_.setBxMaskDefault(0);
  

  // write the condition format to the event setup via the helper:
  using namespace edm::es;
  boost::shared_ptr<L1TGlobalPrescalesVetos> pMenu = boost::shared_ptr< L1TGlobalPrescalesVetos >(data_.getWriteInstance());
  return pMenu;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TGlobalPrescalesVetosESProducer);
