#ifndef HeterogeneousCore_SonicCore_test_DummyClient
#define HeterogeneousCore_SonicCore_test_DummyClient

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/SonicCore/interface/SonicClient.h"

#include <vector>
#include <thread>
#include <chrono>

class DummyClient : public SonicClient<int> {
public:
  //constructor
  DummyClient(const edm::ParameterSet& params, const std::string& debugName)
      : SonicClient<int>(params, debugName, "DummyClient"),
        factor_(params.getParameter<int>("factor")),
        wait_(params.getParameter<int>("wait")),
        fails_(params.getParameter<unsigned>("fails")) {}

  //for fillDescriptions
  static void fillPSetDescription(edm::ParameterSetDescription& iDesc) {
    edm::ParameterSetDescription descClient;
    fillBasePSetDescription(descClient);
    descClient.add<int>("factor", -1);
    descClient.add<int>("wait", 10);
    descClient.add<unsigned>("fails", 0);
    iDesc.add<edm::ParameterSetDescription>("Client", descClient);
  }

protected:
  void evaluate() override {
    //simulate a blocking call
    std::this_thread::sleep_for(std::chrono::seconds(wait_));

    this->output_ = this->input_ * factor_;

    //simulate a failure
    if (this->tries_ < fails_)
      this->finish(false);
    else
      this->finish(true);
  }

  //members
  int factor_;
  int wait_;
  unsigned fails_;
};

#endif
