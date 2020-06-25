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

template <typename Client>
class TritonImageProducer : public SonicEDProducer<Client> {
public:
  //needed because base class has dependent scope
  using typename SonicEDProducer<Client>::Input;
  using typename SonicEDProducer<Client>::Output;
  explicit TritonImageProducer(edm::ParameterSet const& cfg)
      : SonicEDProducer<Client>(cfg), topN_(cfg.getParameter<unsigned>("topN")) {
    //for debugging
    this->setDebugName("TritonImageProducer");
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
    iInput.resize(client_.nInput() * client_.batchSize(), 0.5f);
  }
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) override {
    //check the results
    findTopN(iOutput);
  }
  ~TritonImageProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    Client::fillPSetDescription(desc);
    desc.add<unsigned>("topN", 5);
    desc.add<std::string>("imageList");
    //to ensure distinct cfi names
    descriptions.addWithDefaultLabel(desc);
  }

private:
  using SonicEDProducer<Client>::client_;
  void findTopN(const std::vector<float>& scores, unsigned n = 5) const {
    auto dim = client_.nOutput();
    for (unsigned i0 = 0; i0 < client_.batchSize(); i0++) {
      //match score to type by index, then put in largest-first map
      std::map<float, std::string, std::greater<float>> score_map;
      for (unsigned i = 0; i < std::min((unsigned)dim, (unsigned)imageList_.size()); ++i) {
        score_map.emplace(scores[i0 * dim + i], imageList_[i]);
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

using TritonImageProducerSync = TritonImageProducer<TritonClientSync>;
using TritonImageProducerAsync = TritonImageProducer<TritonClientAsync>;
using TritonImageProducerPseudoAsync = TritonImageProducer<TritonClientPseudoAsync>;

DEFINE_FWK_MODULE(TritonImageProducerSync);
DEFINE_FWK_MODULE(TritonImageProducerAsync);
DEFINE_FWK_MODULE(TritonImageProducerPseudoAsync);
