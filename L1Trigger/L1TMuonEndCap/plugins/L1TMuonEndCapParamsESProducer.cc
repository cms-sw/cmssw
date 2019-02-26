#include <iostream>
#include <memory>
#include <iostream>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducts.h"

#include "CondFormats/L1TObjects/interface/L1TMuonEndCapParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndCapParamsRcd.h"
#include "L1Trigger/L1TMuonEndCap/interface/EndCapParamsHelper.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "TXMLEngine.h"

using namespace std;

// Class declaration

class L1TMuonEndCapParamsESProducer : public edm::ESProducer {
public:
  L1TMuonEndCapParamsESProducer(const edm::ParameterSet&);
  ~L1TMuonEndCapParamsESProducer() override;

  using ReturnType = std::unique_ptr<L1TMuonEndCapParams>;

  ReturnType produce(const L1TMuonEndCapParamsRcd&);

private:
  l1t::EndCapParamsHelper data_;
};

// Constructor

L1TMuonEndCapParamsESProducer::L1TMuonEndCapParamsESProducer(const edm::ParameterSet& iConfig) :
  data_(new L1TMuonEndCapParams())
{
  // The following line is needed to tell the framework what data is being produced
   setWhatProduced(this);

   data_.SetPtAssignVersion(iConfig.getParameter<int>("PtAssignVersion"));
   data_.SetFirmwareVersion(iConfig.getParameter<int>("FirmwareVersion"));
   data_.SetPrimConvVersion(iConfig.getParameter<int>("PrimConvVersion"));

}

// Destructor

L1TMuonEndCapParamsESProducer::~L1TMuonEndCapParamsESProducer()
{
}

// Member functions

// ------------ method called to produce the data  ------------
L1TMuonEndCapParamsESProducer::ReturnType
L1TMuonEndCapParamsESProducer::produce(const L1TMuonEndCapParamsRcd& iRecord)
{
   return std::make_unique<L1TMuonEndCapParams>(*data_.getWriteInstance());
}

// Define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonEndCapParamsESProducer);
