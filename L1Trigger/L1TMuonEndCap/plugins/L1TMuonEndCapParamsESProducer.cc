#include <iostream>
#include <memory>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducts.h"

#include "CondFormats/L1TObjects/interface/L1TMuonEndCapParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndcapParamsRcd.h"

using namespace std;

// class declaration

class L1TMuonEndCapParamsESProducer : public edm::ESProducer {
public:
  L1TMuonEndCapParamsESProducer(const edm::ParameterSet&);
  ~L1TMuonEndCapParamsESProducer() {}

  typedef std::shared_ptr<L1TMuonEndCapParams> ReturnType;

  ReturnType produce(const L1TMuonEndcapParamsRcd&);

private:
  L1TMuonEndCapParams params;
};

// constructor

L1TMuonEndCapParamsESProducer::L1TMuonEndCapParamsESProducer(const edm::ParameterSet& iConfig)
{
   setWhatProduced(this);

   params.PtAssignVersion_   = iConfig.getParameter<int>("PtAssignVersion");
   params.firmwareVersion_   = iConfig.getParameter<int>("firmwareVersion");
   // Not yet implemented in O2O
   // params.PrimConvVersion_   = iConfig.getParameter<int>("PrimConvVersion");

}


// member functions

L1TMuonEndCapParamsESProducer::ReturnType
L1TMuonEndCapParamsESProducer::produce(const L1TMuonEndcapParamsRcd& iRecord)
{
   std::shared_ptr<L1TMuonEndCapParams> pEMTFParams(&params);
   return pEMTFParams;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonEndCapParamsESProducer);
