#ifndef HeterogeneousCore_SonicCore_test_DummyClient
#define HeterogeneousCore_SonicCore_test_DummyClient

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/SonicCore/interface/SonicClientSync.h"
#include "HeterogeneousCore/SonicCore/interface/SonicClientPseudoAsync.h"
#include "HeterogeneousCore/SonicCore/interface/SonicClientAsync.h"

#include <vector>
#include <thread>
#include <chrono>

template <typename Client>
class DummyClient : public Client {
public:
  //constructor
  DummyClient(const edm::ParameterSet& params)
      : factor_(params.getParameter<int>("factor")), wait_(params.getParameter<int>("wait")) {}

  //for fillDescriptions
  static void fillPSetDescription(edm::ParameterSetDescription& iDesc) {
    edm::ParameterSetDescription descClient;
    descClient.add<int>("factor", -1);
    descClient.add<int>("wait", 10);
    iDesc.add<edm::ParameterSetDescription>("Client", descClient);
  }

protected:
  void evaluate() override {
    //simulate a blocking call
    std::this_thread::sleep_for(std::chrono::seconds(wait_));

    this->output_ = this->input_ * factor_;
    this->finish();
  }

  //members
  int factor_;
  int wait_;
};

typedef DummyClient<SonicClientSync<int>> DummyClientSync;
typedef DummyClient<SonicClientPseudoAsync<int>> DummyClientPseudoAsync;
typedef DummyClient<SonicClientAsync<int>> DummyClientAsync;

#endif
