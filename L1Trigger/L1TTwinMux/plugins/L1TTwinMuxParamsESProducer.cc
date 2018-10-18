// -*- C++ -*-
//
// Class:      L1TTwinMuxParamsESProducer
//
// Original Author:  Giannis Flouris
//         Created:
//
//modifications: g karathanasis


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
//#include "FWCore/Framework/interface/ESHandle.h"
//#include "FWCore/Framework/interface/ESProducts.h"

#include "CondFormats/L1TObjects/interface/L1TTwinMuxParams.h"
#include "CondFormats/DataRecord/interface/L1TTwinMuxParamsRcd.h"


// class declaration
//
typedef std::map<short, short, std::less<short> > LUT;

class L1TTwinMuxParamsESProducer : public edm::ESProducer {
   public:
      L1TTwinMuxParamsESProducer(const edm::ParameterSet&);
      ~L1TTwinMuxParamsESProducer() override;

      using ReturnType = std::unique_ptr<L1TTwinMuxParams>;

      ReturnType produce(const L1TTwinMuxParamsRcd&);
   private:
      L1TTwinMuxParams m_params;
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
L1TTwinMuxParamsESProducer::L1TTwinMuxParamsESProducer(const edm::ParameterSet& iConfig)
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


L1TTwinMuxParamsESProducer::~L1TTwinMuxParamsESProducer()
{

}


//
// member functions
//

// ------------ method called to produce the data  ------------
L1TTwinMuxParamsESProducer::ReturnType
L1TTwinMuxParamsESProducer::produce(const L1TTwinMuxParamsRcd& iRecord)
{
   return std::make_unique<L1TTwinMuxParams>(m_params);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TTwinMuxParamsESProducer);
