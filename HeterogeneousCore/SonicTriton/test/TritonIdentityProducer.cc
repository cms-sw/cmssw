#include "HeterogeneousCore/SonicTriton/interface/TritonEDProducer.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>

class TritonIdentityProducer : public TritonEDProducer<> {
public:
  explicit TritonIdentityProducer(edm::ParameterSet const& cfg)
      : TritonEDProducer<>(cfg), batchSizes_{1, 2, 0}, batchCounter_(0) {}
  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) override {
    //follow Triton QA tests for ragged input
    std::vector<std::vector<float>> value_lists{{2, 2}, {4, 4, 4, 4}, {1}, {3, 3, 3}};

    client_->setBatchSize(batchSizes_[batchCounter_]);
    batchCounter_ = (batchCounter_ + 1) % batchSizes_.size();
    auto& input1 = iInput.at("INPUT0");
    auto data1 = input1.allocate<float>();
    for (unsigned i = 0; i < client_->batchSize(); ++i) {
      (*data1)[i] = value_lists[i];
      input1.setShape(0, (*data1)[i].size(), i);
    }

    // convert to server format
    input1.toServer(data1);
  }
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) override {
    // check the results
    const auto& output1 = iOutput.at("OUTPUT0");
    // convert from server format
    const auto& tmp = output1.fromServer<float>();
    edm::LogInfo msg(debugName_);
    for (unsigned i = 0; i < client_->batchSize(); ++i) {
      msg << "output " << i << " (" << triton_utils::printColl(output1.shape(i)) << "): ";
      for (int j = 0; j < output1.shape(i)[0]; ++j) {
        msg << tmp[i][j] << " ";
      }
      msg << "\n";
    }
  }
  ~TritonIdentityProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    TritonClient::fillPSetDescription(desc);
    //to ensure distinct cfi names
    descriptions.addWithDefaultLabel(desc);
  }

private:
  std::vector<unsigned> batchSizes_;
  unsigned batchCounter_;
};

DEFINE_FWK_MODULE(TritonIdentityProducer);
