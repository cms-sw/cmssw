#include "HeterogeneousCore/SonicCore/interface/SonicEDProducer.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonClient.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class TritonImageProducer : public SonicEDProducer<TritonClient> {
public:
  explicit TritonImageProducer(edm::ParameterSet const& cfg)
      : SonicEDProducer<TritonClient>(cfg), topN_(cfg.getParameter<unsigned>("topN")) {
    //for debugging
    setDebugName("TritonImageProducer");
    //load score list
    std::string imageListFile(cfg.getParameter<std::string>("imageList"));
    std::ifstream ifile(imageListFile);
    if (ifile.is_open()) {
      std::string line;
      while (std::getline(ifile, line)) {
        imageList_.push_back(line);
      }
    } else {
      throw cms::Exception("MissingFile") << "Could not open image list file: " << imageListFile;
    }
  }
  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) override {
    // create an npix x npix x ncol image w/ arbitrary color value
    // model only has one input, so just pick begin()
    auto& input1 = iInput.begin()->second;
    auto data1 = std::make_shared<TritonInput<float>>();
    data1->reserve(input1.batchSize());
    for (unsigned i = 0; i < input1.batchSize(); ++i) {
      data1->emplace_back(input1.sizeDims(), 0.5f);
    }
    // convert to server format
    input1.toServer(data1);
  }
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) override {
    // check the results
    findTopN(iOutput.begin()->second);
  }
  ~TritonImageProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    TritonClient::fillPSetDescription(desc);
    desc.add<unsigned>("topN", 5);
    desc.add<std::string>("imageList");
    //to ensure distinct cfi names
    descriptions.addWithDefaultLabel(desc);
  }

private:
  void findTopN(const TritonOutputData& scores, unsigned n = 5) const {
    const auto& tmp = scores.fromServer<float>();
    auto dim = scores.sizeDims();
    for (unsigned i0 = 0; i0 < scores.batchSize(); i0++) {
      //match score to type by index, then put in largest-first map
      std::map<float, std::string, std::greater<float>> score_map;
      for (unsigned i = 0; i < std::min((unsigned)dim, (unsigned)imageList_.size()); ++i) {
        score_map.emplace(tmp[i0][i], imageList_[i]);
      }
      //get top n
      std::stringstream msg;
      msg << "Scores for image " << i0 << ":\n";
      unsigned counter = 0;
      for (const auto& item : score_map) {
        msg << item.second << " : " << item.first << "\n";
        ++counter;
        if (counter >= topN_)
          break;
      }
      edm::LogInfo(client_.debugName()) << msg.str();
    }
  }

  unsigned topN_;
  std::vector<std::string> imageList_;
};

DEFINE_FWK_MODULE(TritonImageProducer);
