#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

class SiStripFedCabling;

class LaserAlignmentEventFilter : public edm::global::EDFilter<> {

public:
  explicit LaserAlignmentEventFilter(const edm::ParameterSet&);
  ~LaserAlignmentEventFilter();

private:
  virtual bool filter(edm::StreamID, edm::Event &, edm::EventSetup const&) const override;

  // FED RAW data input collection
  const edm::EDGetTokenT<FEDRawDataCollection> FED_collection_token;

  // filter settings
  const std::vector<uint16_t> las_fed_ids;      // list of FEDs used by LAS
  const std::vector<uint32_t> las_signal_ids;   // list of DetIds to probe for signal

  const uint16_t single_channel_thresh;         // signal threshold for a single channel
  const uint16_t channel_count_thresh;          // nr. of channels that have to contain signal for LAS event
};
