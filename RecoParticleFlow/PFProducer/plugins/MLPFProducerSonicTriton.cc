#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <map>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "HeterogeneousCore/SonicCore/interface/SonicEDProducer.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonClient.h"
#include "RecoParticleFlow/PFProducer/interface/MLPFModel.h"

class MLPFProducerSonicTriton : public SonicEDProducer<TritonClient> {
public:
  explicit MLPFProducerSonicTriton(edm::ParameterSet const& cfg)
      : SonicEDProducer<TritonClient>(cfg),
        pfCandidatesPutToken_{produces<reco::PFCandidateCollection>()},
        inputTagBlocks_(consumes<reco::PFBlockCollection>(cfg.getParameter<edm::InputTag>("src"))) {
    this->setDebugName("MLPFProducerSonic");
  }

  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) override {
    using namespace reco::mlpf;

    //get the PFElements in the event. Currently use PFBlock for convenience, but we don't need anything
    //else the block does, later we can get the events directly from the event.
    const auto& blocks = iEvent.get(inputTagBlocks_);
    const auto& all_elements = getPFElements(blocks);

    const auto num_elements_total = all_elements.size();

    //tensor size must be a multiple of the bin size and larger than the number of elements
    const auto tensor_size = LSH_BIN_SIZE * (num_elements_total / LSH_BIN_SIZE + 1);
    assert(tensor_size <= NUM_MAX_ELEMENTS_BATCH);

    auto& input1 = iInput.at("x");

    //we ignore Sonic/Triton batching, as it doesn't create a dim-3 input for batch size 1.
    //instead, we specify the batch dim as a model dim.
    input1.setShape(0, 1);
    input1.setShape(1, tensor_size);

    auto data1 = std::make_shared<TritonInput<float>>(1);
    auto& vdata1 = (*data1)[0];
    vdata1.reserve(input1.sizeShape());

    //Fill the input tensor
    for (const auto* pelem : all_elements) {
      const auto& elem = *pelem;

      //prepare the input array from the PFElement
      const auto& props = getElementProperties(elem);

      //copy features to the input array
      for (unsigned int iprop = 0; iprop < NUM_ELEMENT_FEATURES; iprop++) {
        vdata1.push_back(normalize(props[iprop]));
      }
    }
    input1.toServer(data1);
  }

  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) override {
    using namespace reco::mlpf;

    //we need the input element list to set the refs on the produced candidate. Currently use PFBlock for convenience, but we don't need anything
    //else the block does, later we can get the events directly from the event.
    const auto& blocks = iEvent.get(inputTagBlocks_);
    const auto& all_elements = getPFElements(blocks);

    std::vector<reco::PFCandidate> pOutputCandidateCollection;
    const auto& output1 = iOutput.at("Identity");

    //get the data of the first (and only) batch
    const auto& out_data = output1.fromServer<float>();

    //batch size 1
    assert(output1.shape()[0] == 1);

    //model should have the correct number of outputs
    assert(output1.shape()[2] == NUM_OUTPUTS);

    //we process only uyp to the true number of input elements, the predicion is padded to the bin size
    const auto num_elem = all_elements.size();

    for (size_t ielem = 0; ielem < num_elem; ielem++) {
      //get the coefficients in the output corresponding to the class probabilities (raw logits)
      std::vector<float> pred_id_logits;
      for (unsigned int idx_id = 0; idx_id <= NUM_CLASS; idx_id++) {
        pred_id_logits.push_back(out_data[0][ielem * NUM_OUTPUTS + idx_id]);
      }

      //get the most probable class PDGID
      int pred_pid = pdgid_encoding[argMax(pred_id_logits)];

      //get the predicted momentum components
      float pred_eta = out_data[0][ielem * NUM_OUTPUTS + IDX_ETA];
      float pred_phi = out_data[0][ielem * NUM_OUTPUTS + IDX_PHI];
      float pred_e = out_data[0][ielem * NUM_OUTPUTS + IDX_ENERGY];
      float pred_charge = out_data[0][ielem * NUM_OUTPUTS + IDX_CHARGE];

      //a particle was predicted for this PFElement, otherwise it was a spectator
      if (pred_pid != 0) {
        auto cand = makeCandidate(pred_pid, pred_charge, pred_e, pred_eta, pred_phi);
        setCandidateRefs(cand, all_elements, ielem);
        pOutputCandidateCollection.push_back(cand);
      }
    }  //loop over PFElements

    iEvent.emplace(pfCandidatesPutToken_, pOutputCandidateCollection);
  }

  ~MLPFProducerSonicTriton() override {}

  //to ensure distinct cfi names - specialized below
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    TritonClient::fillPSetDescription(desc);
    desc.add<edm::InputTag>("src", edm::InputTag("particleFlowBlock"));
    descriptions.addWithDefaultLabel(desc);
  }

private:
  const edm::EDPutTokenT<reco::PFCandidateCollection> pfCandidatesPutToken_;
  const edm::EDGetTokenT<reco::PFBlockCollection> inputTagBlocks_;
};

DEFINE_FWK_MODULE(MLPFProducerSonicTriton);