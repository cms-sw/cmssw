// -*- C++ -*-
//
// Class:      L1TwinMuxParamsESProducer
//
// Original Author:  Giannis Flouris
//         Created:
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducts.h"

#include "CondFormats/L1TObjects/interface/L1TwinMuxParams.h"
#include "CondFormats/DataRecord/interface/L1TwinMuxParamsRcd.h"


// class declaration
//
typedef std::map<short, short, std::less<short> > LUT;

class L1TwinMuxParamsESProducer : public edm::ESProducer {
   public:
      L1TwinMuxParamsESProducer(const edm::ParameterSet&);
      ~L1TwinMuxParamsESProducer();

      typedef std::shared_ptr<L1TwinMuxParams> ReturnType;

      ReturnType produce(const L1TwinMuxParamsRcd&);
   private:
      L1TwinMuxParams m_params;
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
L1TwinMuxParamsESProducer::L1TwinMuxParamsESProducer(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);
   // Firmware version
   unsigned fwVersion = iConfig.getParameter<unsigned>("fwVersion");
   unsigned useRpcBxForDtBelowQuality = iConfig.getParameter<unsigned>("useRpcBxForDtBelowQuality");
   bool     useOnlyRPC = iConfig.getParameter<bool>("useOnlyRPC");
   bool     useOnlyDT = iConfig.getParameter<bool>("useOnlyDT");
   bool     useLowQDT = iConfig.getParameter<bool>("useLowQDT");
   bool     CorrectDTBxwRPC = iConfig.getParameter<bool>("CorrectDTBxwRPC");
   bool     Verbose = iConfig.getParameter<bool>("verbose");
   unsigned      dphiWindowBxShift = iConfig.getParameter<unsigned>("dphiWindowBxShift");

   m_params.setFwVersion(fwVersion);
   m_params.set_USERPCBXFORDTBELOWQUALITY(useRpcBxForDtBelowQuality);
   m_params.set_UseOnlyRPC(useOnlyRPC);
   m_params.set_UseOnlyDT(useOnlyDT);
   m_params.set_UseLowQDT(useLowQDT);
   m_params.set_CorrectDTBxwRPC(CorrectDTBxwRPC);
   m_params.set_Verbose(Verbose);
   m_params.set_DphiWindowBxShift(dphiWindowBxShift);



}


L1TwinMuxParamsESProducer::~L1TwinMuxParamsESProducer()
{

}


//
// member functions
//

// ------------ method called to produce the data  ------------
L1TwinMuxParamsESProducer::ReturnType
L1TwinMuxParamsESProducer::produce(const L1TwinMuxParamsRcd& iRecord)
{
   using namespace edm::es;
   // boost::shared_ptr<L1TwinMuxParams> pTMParams;

   // pTMParams = boost::shared_ptr<L1TwinMuxParams>(new L1TwinMuxParams(m_params));
   //return pTMParams;
   return std::make_shared<L1TwinMuxParams>(m_params);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TwinMuxParamsESProducer);
