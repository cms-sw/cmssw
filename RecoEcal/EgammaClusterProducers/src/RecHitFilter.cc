/** \class RecHitFilter
 **   simple filter of EcalRecHits
 **
 **  \author Shahram Rahatlou, University of Rome & INFN, May 2006
 **
 ***/

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <memory>
#include <vector>

class RecHitFilter : public edm::global::EDProducer<> {
public:
  RecHitFilter(const edm::ParameterSet& ps);

  ~RecHitFilter() override;

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const double noiseEnergyThreshold_;
  const double noiseChi2Threshold_;
  const std::string reducedHitCollection_;
  const edm::EDGetTokenT<EcalRecHitCollection> hitCollection_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecHitFilter);

RecHitFilter::RecHitFilter(const edm::ParameterSet& ps)
    : noiseEnergyThreshold_(ps.getParameter<double>("noiseEnergyThreshold")),
      noiseChi2Threshold_(ps.getParameter<double>("noiseChi2Threshold")),
      reducedHitCollection_(ps.getParameter<std::string>("reducedHitCollection")),
      hitCollection_(consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("hitCollection"))) {
  produces<EcalRecHitCollection>(reducedHitCollection_);
}

RecHitFilter::~RecHitFilter() {}

void RecHitFilter::produce(edm::StreamID, edm::Event& evt, const edm::EventSetup& es) const {
  // get the hit collection from the event:
  edm::Handle<EcalRecHitCollection> rhcHandle;
  evt.getByToken(hitCollection_, rhcHandle);
  const EcalRecHitCollection* hit_collection = rhcHandle.product();

  int nTot = hit_collection->size();
  int nRed = 0;

  // create a unique_ptr to a BasicClusterCollection, copy the clusters into it and put in the Event:
  auto redCollection = std::make_unique<EcalRecHitCollection>();

  for (EcalRecHitCollection::const_iterator it = hit_collection->begin(); it != hit_collection->end(); ++it) {
    //std::cout << *it << std::endl;
    if (it->energy() > noiseEnergyThreshold_ && it->chi2() < noiseChi2Threshold_) {
      nRed++;
      redCollection->push_back(EcalRecHit(*it));
    }
  }

  edm::LogInfo("") << "total # hits: " << nTot << "  #hits with E > " << noiseEnergyThreshold_ << " GeV  and  chi2 < "
                   << noiseChi2Threshold_ << " : " << nRed << std::endl;

  evt.put(std::move(redCollection), reducedHitCollection_);
}
