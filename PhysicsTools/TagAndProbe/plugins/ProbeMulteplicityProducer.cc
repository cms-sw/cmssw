//

/**
  \class    ProbeMulteplicityProducer"
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
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

class ProbeMulteplicityProducer : public edm::stream::EDProducer<> {
public:
  explicit ProbeMulteplicityProducer(const edm::ParameterSet& iConfig);
  ~ProbeMulteplicityProducer() override;

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  edm::EDGetTokenT<reco::CandidateView> pairs_;
  StringCutObjectSelector<reco::Candidate, true>
      pairCut_;  // lazy parsing, to allow cutting on variables not in reco::Candidate class
  StringCutObjectSelector<reco::Candidate, true>
      probeCut_;  // lazy parsing, to allow cutting on variables not in reco::Candidate class
};

ProbeMulteplicityProducer::ProbeMulteplicityProducer(const edm::ParameterSet& iConfig)
    : pairs_(consumes<reco::CandidateView>(iConfig.getParameter<edm::InputTag>("pairs"))),
      pairCut_(iConfig.existsAs<std::string>("pairSelection") ? iConfig.getParameter<std::string>("pairSelection") : "",
               true),
      probeCut_(
          iConfig.existsAs<std::string>("probeSelection") ? iConfig.getParameter<std::string>("probeSelection") : "",
          true) {
  produces<edm::ValueMap<float>>();
}

ProbeMulteplicityProducer::~ProbeMulteplicityProducer() {}

void ProbeMulteplicityProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // read input
  Handle<View<reco::Candidate>> pairs;
  iEvent.getByToken(pairs_, pairs);

  // fill
  unsigned int i = 0;
  std::vector<unsigned int> tagKeys;
  std::vector<float> values;
  View<reco::Candidate>::const_iterator pair, endpairs = pairs->end();
  for (pair = pairs->begin(); pair != endpairs; ++pair, ++i) {
    reco::CandidateBaseRef probeRef = pair->daughter(1)->masterClone();
    unsigned int tagKey = pair->daughter(0)->masterClone().key();
    unsigned int copies = 1;
    if (pairCut_(*pair) && probeCut_(*probeRef)) {
      for (unsigned int j = 0; j < i; ++j)
        if (tagKeys[j] == tagKey)
          copies++;
      for (unsigned int j = 0; j < i; ++j)
        if (tagKeys[j] == tagKey)
          values[j] = copies;
    } else {
      tagKey = std::numeric_limits<unsigned int>::max();
      copies = 0;
    }
    tagKeys.push_back(tagKey);
    values.push_back(copies);
  }

  // convert into ValueMap and store
  auto valMap = std::make_unique<ValueMap<float>>();
  ValueMap<float>::Filler filler(*valMap);
  filler.insert(pairs, values.begin(), values.end());
  filler.fill();
  iEvent.put(std::move(valMap));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ProbeMulteplicityProducer);
