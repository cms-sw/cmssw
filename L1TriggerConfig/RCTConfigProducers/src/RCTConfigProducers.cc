// -*- C++ -*-
//
// Package:    RCTConfigProducers
// Class:      RCTConfigProducers
// 
/**\class RCTConfigProducers RCTConfigProducers.h L1TriggerConfig/RCTConfigProducers/src/RCTConfigProducers.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sridhara Dasu
//         Created:  Mon Jul 16 23:48:35 CEST 2007
// $Id: RCTConfigProducers.cc,v 1.4 2007/07/31 08:39:41 dasu Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1RCTParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"

//
// class decleration
//

class RCTConfigProducers : public edm::ESProducer {
   public:
      RCTConfigProducers(const edm::ParameterSet&);
      ~RCTConfigProducers();

      typedef boost::shared_ptr<L1RCTParameters> ReturnType;

      ReturnType produce(const L1RCTParametersRcd&);
private:
      // ----------member data ---------------------------
  L1RCTParameters *rctParameters;
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
RCTConfigProducers::RCTConfigProducers(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);
   //now do what ever other initialization is needed
   rctParameters = 
     new L1RCTParameters(iConfig.getParameter<double>("eGammaLSB"),
			 iConfig.getParameter<double>("jetMETLSB"),
			 iConfig.getParameter<double>("eMinForFGCut"),
			 iConfig.getParameter<double>("eMaxForFGCut"),
			 iConfig.getParameter<double>("hOeCut"),
			 iConfig.getParameter<double>("eMinForHoECut"),
			 iConfig.getParameter<double>("eMaxForHoECut"),
			 iConfig.getParameter<double>("eActivityCut"),
			 iConfig.getParameter<double>("hActivityCut"),
			 iConfig.getParameter<std::vector< double > >("eGammaECalScaleFactors"),
                         iConfig.getParameter<std::vector< double > >("eGammaHCalScaleFactors"),
                         iConfig.getParameter<std::vector< double > >("jetMETECalScaleFactors"),
                         iConfig.getParameter<std::vector< double > >("jetMETHCalScaleFactors")
			 );
}


RCTConfigProducers::~RCTConfigProducers()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
RCTConfigProducers::ReturnType
RCTConfigProducers::produce(const L1RCTParametersRcd& iRecord)
{
   using namespace edm::es;
   boost::shared_ptr<L1RCTParameters> pL1RCTParameters =
     (boost::shared_ptr<L1RCTParameters>) rctParameters;
   return pL1RCTParameters ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(RCTConfigProducers);
