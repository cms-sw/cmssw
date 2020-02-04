#include "LaserAlignmentEventFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"

#include <algorithm>

// make a sorted copy of a vector, optionally of a different type
template <typename T, typename U>
std::vector<T> copy_and_sort_vector(std::vector<U> const& input) {
  std::vector<T> copy(input.begin(), input.end());
  std::sort(copy.begin(), copy.end());
  return copy;
}

///
/// constructors and destructor
///
LaserAlignmentEventFilter::LaserAlignmentEventFilter(const edm::ParameterSet& iConfig)
    : FED_collection_token(consumes<FEDRawDataCollection>(iConfig.getParameter<edm::InputTag>("FedInputTag"))),
      las_fed_ids(copy_and_sort_vector<uint16_t>(iConfig.getParameter<std::vector<int>>("FED_IDs"))),
      las_signal_ids(copy_and_sort_vector<uint32_t>(iConfig.getParameter<std::vector<int>>("SIGNAL_IDs"))),
      single_channel_thresh(iConfig.getParameter<unsigned>("SINGLE_CHANNEL_THRESH")),
      channel_count_thresh(iConfig.getParameter<unsigned>("CHANNEL_COUNT_THRESH")) {}

///
///
///
LaserAlignmentEventFilter::~LaserAlignmentEventFilter() {}

/// Checks for Laser Signals in specific modules
/// For Accessing FED Data see also EventFilter/SiStripRawToDigi/src/SiStripRawToDigiUnpacker.cc and related files
bool LaserAlignmentEventFilter::filter(edm::StreamID sid, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  //unsigned int  det_ctr = 0;      // count how many modules are tested for signal
  unsigned int sig_ctr = 0;  // count how many modules have signal
  //unsigned long buffer_sum = 0;   // sum of buffer sizes

  // retrieve FED raw data (by label, which is "source" by default)
  edm::Handle<FEDRawDataCollection> buffers;
  iEvent.getByToken(FED_collection_token, buffers);

  // read the cabling map from the EventSetup
  edm::ESHandle<SiStripFedCabling> cabling;
  iSetup.get<SiStripFedCablingRcd>().get(cabling);

  std::vector<uint16_t>::const_iterator ifed = las_fed_ids.begin();
  for (; ifed != las_fed_ids.end(); ifed++) {
    // Retrieve FED raw data for given FED
    const FEDRawData& input = buffers->FEDData(static_cast<int>(*ifed));
    LogDebug("LaserAlignmentEventFilter") << "Examining FED " << *ifed;

    // check on FEDRawData pointer
    if (not input.data())
      continue;

    // check on FEDRawData size
    if (not input.size())
      continue;

    // construct FEDBuffer
    const auto st_buffer = sistrip::preconstructCheckFEDBuffer(input);
    if (sistrip::FEDBufferStatusCode::SUCCESS != st_buffer) {
      throw cms::Exception("FEDBuffer") << st_buffer << " (check debug output for more details)";
    }
    sistrip::FEDBuffer buffer{input};
    const auto st_chan = buffer.findChannels();
    if (sistrip::FEDBufferStatusCode::SUCCESS != st_chan) {
      throw cms::Exception("FEDBuffer") << st_chan << " (check debug output for more details)";
    }
    if (not buffer.doChecks(true)) {
      edm::LogWarning("LaserAlignmentEventFilter") << "FED Buffer check fails for FED ID " << *ifed << ".";
      continue;
    }

    // get the cabling connections for this FED
    auto const& conns = cabling->fedConnections(*ifed);

    // Iterate through FED channels, extract payload and create Digis
    std::vector<FedChannelConnection>::const_iterator iconn = conns.begin();
    for (; iconn != conns.end(); iconn++) {
      if (std::binary_search(las_signal_ids.begin(), las_signal_ids.end(), iconn->detId())) {
        LogDebug("LaserAlignmentEventFilter")
            << " Found LAS signal module in FED " << *ifed << "  DetId: " << iconn->detId() << "\n"
            << "buffer.channel(iconn->fedCh()).size(): " << buffer.channel(iconn->fedCh()).length();
        //++det_ctr;
        if (buffer.channel(iconn->fedCh()).length() > single_channel_thresh)
          ++sig_ctr;
        //buffer_sum += buffer.channel(iconn->fedCh()).length();
        if (sig_ctr > channel_count_thresh) {
          LogDebug("LaserAlignmentEventFilter") << "Event identified as LAS";
          return true;
        }
      }
    }  // channel loop
  }    // FED loop

  //   LogDebug("LaserAlignmentEventFilter") << det_ctr << " channels were tested for signal\n"
  // 			    <<sig_ctr << " channels have signal\n"
  // 			    << "Sum of buffer sizes: " << buffer_sum;

  return false;
}

//define this as a plug-in
DEFINE_FWK_MODULE(LaserAlignmentEventFilter);
