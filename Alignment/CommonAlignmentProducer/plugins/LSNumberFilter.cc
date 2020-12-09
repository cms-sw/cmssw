#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class declaration
//

class LSNumberFilter : public edm::stream::EDFilter<> {
public:
  explicit LSNumberFilter(const edm::ParameterSet&);
  ~LSNumberFilter() override;

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;
  unsigned int minLS;
};

LSNumberFilter::LSNumberFilter(const edm::ParameterSet& iConfig)
    : minLS(iConfig.getUntrackedParameter<unsigned>("minLS", 21)) {}

LSNumberFilter::~LSNumberFilter() {}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool LSNumberFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (iEvent.luminosityBlock() < minLS)
    return false;

  return true;
}

//define this as a plug-in
DEFINE_FWK_MODULE(LSNumberFilter);
