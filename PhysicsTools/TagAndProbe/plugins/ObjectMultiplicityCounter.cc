//

/**
  \class    ObjectMultiplicityCounter"
  \brief    Matcher of number of reconstructed objects in the event to probe

  \author   Kalanand Mishra
*/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

template <typename T>
class ObjectMultiplicityCounter : public edm::stream::EDProducer<> {
public:
  explicit ObjectMultiplicityCounter(const edm::ParameterSet& iConfig);
  ~ObjectMultiplicityCounter() override;

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  edm::EDGetTokenT<edm::View<reco::Candidate>> probesToken_;
  edm::EDGetTokenT<edm::View<T>> objectsToken_;
  StringCutObjectSelector<T, true> objCut_;  // lazy parsing, to allow cutting on variables not in reco::Candidate class
};

template <typename T>
ObjectMultiplicityCounter<T>::ObjectMultiplicityCounter(const edm::ParameterSet& iConfig)
    : probesToken_(consumes<edm::View<reco::Candidate>>(iConfig.getParameter<edm::InputTag>("probes"))),
      objectsToken_(consumes<edm::View<T>>(iConfig.getParameter<edm::InputTag>("objects"))),
      objCut_(
          iConfig.existsAs<std::string>("objectSelection") ? iConfig.getParameter<std::string>("objectSelection") : "",
          true) {
  produces<edm::ValueMap<float>>();
}

template <typename T>
ObjectMultiplicityCounter<T>::~ObjectMultiplicityCounter() {}

template <typename T>
void ObjectMultiplicityCounter<T>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // read input
  Handle<View<reco::Candidate>> probes;
  Handle<View<T>> objects;
  iEvent.getByToken(probesToken_, probes);
  iEvent.getByToken(objectsToken_, objects);

  // fill
  float count = 0.0;
  typename View<T>::const_iterator object, endobjects = objects->end();
  for (object = objects->begin(); object != endobjects; ++object) {
    if (!(objCut_(*object)))
      continue;
    count += 1.0;
  }

  // prepare vector for output
  std::vector<float> values(probes->size(), count);

  // convert into ValueMap and store
  auto valMap = std::make_unique<ValueMap<float>>();
  ValueMap<float>::Filler filler(*valMap);
  filler.insert(probes, values.begin(), values.end());
  filler.fill();
  iEvent.put(std::move(valMap));
}

typedef ObjectMultiplicityCounter<reco::Candidate> CandMultiplicityCounter;
typedef ObjectMultiplicityCounter<reco::Track> TrackMultiplicityCounter;
typedef ObjectMultiplicityCounter<reco::Vertex> VertexMultiplicityCounter;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CandMultiplicityCounter);
DEFINE_FWK_MODULE(TrackMultiplicityCounter);
DEFINE_FWK_MODULE(VertexMultiplicityCounter);
