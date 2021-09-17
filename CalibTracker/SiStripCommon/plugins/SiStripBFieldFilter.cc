#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

//
// -- Class Deleration
//

class SiStripBFieldFilter : public edm::stream::EDFilter<> {
public:
  SiStripBFieldFilter(const edm::ParameterSet&);
  ~SiStripBFieldFilter() override;

private:
  const edm::ESGetToken<RunInfo, RunInfoRcd> runInfoToken_;
  double MagFieldCurrentTh_; /*!<  Threshold for the Magnet current. */
  bool HIpassFilter_;        /*!<  Swith for HI/Low filter. */

  bool filter(edm::Event&, edm::EventSetup const&) override;
};

//
// -- Constructor
//
SiStripBFieldFilter::SiStripBFieldFilter(const edm::ParameterSet& pset)
    : runInfoToken_(esConsumes<RunInfo, RunInfoRcd>()) {
  MagFieldCurrentTh_ = pset.getUntrackedParameter<double>("MagFieldCurrentTh", 2000.);
  HIpassFilter_ = pset.getUntrackedParameter<bool>("HIpassFilter", true);
}

//
// -- Destructor
//
SiStripBFieldFilter::~SiStripBFieldFilter() {}

bool SiStripBFieldFilter::filter(edm::Event& evt, edm::EventSetup const& es) {
  edm::ESHandle<RunInfo> runInfo = es.getHandle(runInfoToken_);
  double average_current = runInfo.product()->m_avg_current;

  return ((HIpassFilter_) ? average_current > MagFieldCurrentTh_ : average_current < MagFieldCurrentTh_);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripBFieldFilter);
