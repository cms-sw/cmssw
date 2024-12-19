#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/Span.h"

// L1 scouting
#include "DataFormats/L1Scouting/interface/L1ScoutingMuon.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingCalo.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingBMTFStub.h"
#include "DataFormats/L1Scouting/interface/OrbitCollection.h"

#include <vector>
#include <set>

using namespace l1ScoutingRun3;

template <typename T>
class MaskOrbitBx : public edm::stream::EDProducer<> {
public:
  explicit MaskOrbitBx(const edm::ParameterSet&);
  ~MaskOrbitBx() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  std::vector<std::vector<T>> orbitBuffer_;

  // tokens for scouting data
  edm::EDGetTokenT<OrbitCollection<T>> tokenData_;

  // BX to be keep
  edm::EDGetTokenT<std::vector<unsigned>> tokenSelBxs_;

  std::string productLabel_;

  const int NBX = 3565;
};

template <typename T>
MaskOrbitBx<T>::MaskOrbitBx(const edm::ParameterSet& iPSet)
    : tokenData_(consumes<OrbitCollection<T>>(iPSet.getParameter<edm::InputTag>("dataTag"))),
      tokenSelBxs_(consumes<std::vector<unsigned>>(iPSet.getParameter<edm::InputTag>("selectBxs"))),
      productLabel_(iPSet.getParameter<std::string>("productLabel")) {
  // prepare module buffer
  orbitBuffer_ = std::vector<std::vector<T>>(NBX);

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

  int nObjOrbit_ = 0;

  // fill collections with objects
  for (const unsigned& bx : *selBxs) {
    for (const auto& obj : objCollection->bxIterator(bx)) {
      orbitBuffer_[bx].push_back(obj);
      nObjOrbit_++;
    }
  }

  // fill orbit collection and clear the Bx buffer vector
  selectedObjs->fillAndClear(orbitBuffer_, nObjOrbit_);

  // store collections in the event
  iEvent.put(std::move(selectedObjs), productLabel_);

}  // end produce

template <typename T>
void MaskOrbitBx<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

typedef MaskOrbitBx<l1ScoutingRun3::Muon> MaskOrbitBxScoutingMuon;
typedef MaskOrbitBx<l1ScoutingRun3::Jet> MaskOrbitBxScoutingJet;
typedef MaskOrbitBx<l1ScoutingRun3::EGamma> MaskOrbitBxScoutingEGamma;
typedef MaskOrbitBx<l1ScoutingRun3::Tau> MaskOrbitBxScoutingTau;
typedef MaskOrbitBx<l1ScoutingRun3::BxSums> MaskOrbitBxScoutingBxSums;
typedef MaskOrbitBx<l1ScoutingRun3::BMTFStub> MaskOrbitBxScoutingBMTFStub;

DEFINE_FWK_MODULE(MaskOrbitBxScoutingMuon);
DEFINE_FWK_MODULE(MaskOrbitBxScoutingJet);
DEFINE_FWK_MODULE(MaskOrbitBxScoutingEGamma);
DEFINE_FWK_MODULE(MaskOrbitBxScoutingTau);
DEFINE_FWK_MODULE(MaskOrbitBxScoutingBxSums);
DEFINE_FWK_MODULE(MaskOrbitBxScoutingBMTFStub);
