#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "DataFormats/L1Scouting/interface/L1ScoutingMuon.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingCalo.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingBMTFStub.h"
#include "DataFormats/L1Scouting/interface/OrbitCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

template <typename T>
class MaskOrbitBx : public edm::stream::EDProducer<> {
public:
  explicit MaskOrbitBx(const edm::ParameterSet&);
  ~MaskOrbitBx() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  static constexpr int kNBX = 3565;

  // token for scouting data
  edm::EDGetTokenT<OrbitCollection<T>> const tokenData_;

  // BXs to be kept
  edm::EDGetTokenT<std::vector<unsigned>> const tokenSelBxs_;

  std::string const productLabel_;

  std::vector<std::vector<T>> orbitBuffer_;
};

template <typename T>
MaskOrbitBx<T>::MaskOrbitBx(const edm::ParameterSet& iPSet)
    : tokenData_(consumes(iPSet.getParameter<edm::InputTag>("dataTag"))),
      tokenSelBxs_(consumes(iPSet.getParameter<edm::InputTag>("selectBxs"))),
      productLabel_(iPSet.getParameter<std::string>("productLabel")) {
  // prepare module buffer
  orbitBuffer_ = std::vector<std::vector<T>>(kNBX);

  // products
  produces<OrbitCollection<T>>(productLabel_).setBranchAlias(productLabel_ + "OrbitCollection");
}

// ------------ method called for each ORBIT  ------------
template <typename T>
void MaskOrbitBx<T>::produce(edm::Event& iEvent, const edm::EventSetup&) {
  // get selected BXs
  edm::Handle<std::vector<unsigned>> selBxs;
  iEvent.getByToken(tokenSelBxs_, selBxs);

  // get the data
  edm::Handle<OrbitCollection<T>> objCollection;
  iEvent.getByToken(tokenData_, objCollection);

  // prepare new collections
  std::unique_ptr<OrbitCollection<T>> selectedObjs(new OrbitCollection<T>);

  int nObjOrbit = 0;

  // fill collections with objects
  for (auto const bx : *selBxs) {
    auto const& objs = objCollection->bxIterator(bx);
    orbitBuffer_[bx].reserve(objs.size());
    for (auto const& obj : objs) {
      orbitBuffer_[bx].push_back(obj);
      nObjOrbit++;
    }
  }

  // fill orbit collection and clear the Bx buffer vector
  selectedObjs->fillAndClear(orbitBuffer_, nObjOrbit);

  // store collections in the event
  iEvent.put(std::move(selectedObjs), productLabel_);
}

template <typename T>
void MaskOrbitBx<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("dataTag");
  desc.add<edm::InputTag>("selectBxs");
  desc.add<std::string>("productLabel");
  descriptions.addDefault(desc);
}

typedef MaskOrbitBx<l1ScoutingRun3::Muon> MaskOrbitBxScoutingMuon;
typedef MaskOrbitBx<l1ScoutingRun3::Jet> MaskOrbitBxScoutingJet;
typedef MaskOrbitBx<l1ScoutingRun3::EGamma> MaskOrbitBxScoutingEGamma;
typedef MaskOrbitBx<l1ScoutingRun3::Tau> MaskOrbitBxScoutingTau;
typedef MaskOrbitBx<l1ScoutingRun3::BxSums> MaskOrbitBxScoutingBxSums;
typedef MaskOrbitBx<l1ScoutingRun3::BMTFStub> MaskOrbitBxScoutingBMTFStub;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MaskOrbitBxScoutingMuon);
DEFINE_FWK_MODULE(MaskOrbitBxScoutingJet);
DEFINE_FWK_MODULE(MaskOrbitBxScoutingEGamma);
DEFINE_FWK_MODULE(MaskOrbitBxScoutingTau);
DEFINE_FWK_MODULE(MaskOrbitBxScoutingBxSums);
DEFINE_FWK_MODULE(MaskOrbitBxScoutingBMTFStub);
