#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDCSStatus.h"

//
// -- Class Deleration
//

class SiStripDCSFilter : public edm::stream::EDFilter<> {
public:
  SiStripDCSFilter(const edm::ParameterSet&);

private:
  bool filter(edm::Event&, edm::EventSetup const&) override;
  SiStripDCSStatus dcsStatus_;
};

//
// -- Constructor
//
SiStripDCSFilter::SiStripDCSFilter(const edm::ParameterSet& pset) : dcsStatus_{consumesCollector()} {}

bool SiStripDCSFilter::filter(edm::Event& evt, edm::EventSetup const& es) { return (dcsStatus_.getStatus(evt, es)); }

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripDCSFilter);
