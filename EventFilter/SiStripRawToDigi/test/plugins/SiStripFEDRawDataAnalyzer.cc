#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "EventFilter/SiStripRawToDigi/test/plugins/SiStripFEDRawDataAnalyzer.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <sstream>
#include <string>
#include <map>

using namespace sistrip;
using namespace std;

// -----------------------------------------------------------------------------
//
SiStripFEDRawDataAnalyzer::SiStripFEDRawDataAnalyzer(const edm::ParameterSet& pset)
    : label_(pset.getParameter<edm::InputTag>("InputLabel")) {
  LogDebug("SiStripFEDRawDataAnalyzer") << "[SiStripFEDRawDataAnalyzer::" << __func__ << "]"
                                        << "Constructing object...";
  consumes<FEDRawDataCollection>(label_);
}

// -----------------------------------------------------------------------------
//
SiStripFEDRawDataAnalyzer::~SiStripFEDRawDataAnalyzer() {
  LogDebug("SiStripFEDRawDataAnalyzer") << "[SiStripFEDRawDataAnalyzer::" << __func__ << "]"
                                        << " Destructing object...";
}

// -----------------------------------------------------------------------------
//
void SiStripFEDRawDataAnalyzer::beginJob() {
  LogDebug("SiStripFEDRawDataAnalyzer") << "[SiStripFEDRawDataAnalyzer::" << __func__ << "]";
}

// -----------------------------------------------------------------------------
//
void SiStripFEDRawDataAnalyzer::endJob() {
  LogDebug("SiStripFEDRawDataAnalyzer") << "[SiStripFEDRawDataAnalyzer::" << __func__ << "]";
}

// -----------------------------------------------------------------------------
//
void SiStripFEDRawDataAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  // Retrieve FED cabling object
  edm::ESHandle<SiStripFedCabling> cabling;
  setup.get<SiStripFedCablingRcd>().get(cabling);

  // Retrieve FEDRawData collection
  edm::Handle<FEDRawDataCollection> buffers;
  event.getByLabel(label_, buffers);

  // Local cache of fed ids from cabling
  auto fed_ids = cabling->fedIds();

  // Containers storing fed ids and buffer sizes
  Feds trg_feds;  // trigger feds
  Feds trk_feds;  // tracker feds
  Feds non_trk;   // feds from other sub-dets
  Feds in_data;   // feds in data but missing from cabling
  Feds in_cabl;   // feds in cabling but missing from data

  for (uint16_t ifed = 0; ifed <= sistrip::CMS_FED_ID_MAX; ++ifed) {
    const FEDRawData& fed = buffers->FEDData(static_cast<int>(ifed));
    const FEDTrailer* fed_trailer = reinterpret_cast<const FEDTrailer*>(fed.data() + fed.size() - sizeof(FEDTrailer));
    if (fed.size() && fed_trailer->check()) {
      trg_feds.push_back(Fed(ifed, fed.size()));
    } else {
      if (ifed < sistrip::FED_ID_MIN || ifed > sistrip::FED_ID_MAX) {
        if (fed.size()) {
          non_trk.push_back(Fed(ifed, fed.size()));
        }
      } else {
        bool found = find(fed_ids.begin(), fed_ids.end(), ifed) != fed_ids.end();
        if (fed.size()) {
          if (found) {
            trk_feds.push_back(Fed(ifed, fed.size()));
          } else {
            in_data.push_back(Fed(ifed, fed.size()));
          }
        }
        if (found && fed.size() == 0) {
          in_cabl.push_back(Fed(ifed, 0));
        }
      }
    }
  }

  // Trigger FEDs
  if (edm::isDebugEnabled()) {
    stringstream ss;
    ss << "[SiStripFEDRawDataCheck::" << __func__ << "]"
       << " Found " << trg_feds.size() << " trigger FEDs";
    edm::LogVerbatim(mlRawToDigi_) << ss.str();
    ss << " with ids (and buffer sizes [bytes]): ";
    for (uint16_t ifed = 0; ifed < trg_feds.size(); ++ifed) {
      ss << trg_feds[ifed].first << "(" << trg_feds[ifed].second << ") ";
    }
    LogTrace(mlRawToDigi_) << ss.str();
  }

  // Not strip tracker FEDs
  if (edm::isDebugEnabled()) {
    stringstream ss;
    ss << "[SiStripFEDRawDataCheck::" << __func__ << "]"
       << " Found " << non_trk.size() << " non-tracker FEDs from other sub-dets";
    edm::LogVerbatim(mlRawToDigi_) << ss.str();
    ss << " with ids (and buffer sizes [bytes]): ";
    for (uint16_t ifed = 0; ifed < non_trk.size(); ++ifed) {
      ss << non_trk[ifed].first << "(" << non_trk[ifed].second << ") ";
    }
    LogTrace(mlRawToDigi_) << ss.str();
  }

  // Strip tracker FEDs
  if (edm::isDebugEnabled()) {
    stringstream ss;
    ss << "[SiStripFEDRawDataCheck::" << __func__ << "]"
       << " Found " << trk_feds.size() << " strip tracker FEDs";
    edm::LogVerbatim(mlRawToDigi_) << ss.str();
    ss << " with ids (and buffer sizes [bytes]): ";
    for (uint16_t ifed = 0; ifed < trk_feds.size(); ++ifed) {
      ss << trk_feds[ifed].first << "(" << trk_feds[ifed].second << ") ";
    }
    LogTrace(mlRawToDigi_) << ss.str();
  }

  // FEDs in data but missing from cabling
  if (edm::isDebugEnabled()) {
    stringstream ss;
    ss << "[SiStripFEDRawDataCheck::" << __func__ << "]"
       << " Found " << in_data.size() << " FEDs in data but missing from cabling";
    edm::LogVerbatim(mlRawToDigi_) << ss.str();
    ss << " with ids (and buffer sizes [bytes]): ";
    for (uint16_t ifed = 0; ifed < in_data.size(); ++ifed) {
      ss << in_data[ifed].first << "(" << in_data[ifed].second << ") ";
    }
    LogTrace(mlRawToDigi_) << ss.str();
  }

  // FEDs in cabling but missing from data
  if (edm::isDebugEnabled()) {
    stringstream ss;
    ss << "[SiStripFEDRawDataCheck::" << __func__ << "]"
       << " Found " << in_cabl.size() << " FEDs in cabling but missing from data";
    edm::LogVerbatim(mlRawToDigi_) << ss.str();
    ss << " with ids (and zero buffer size): ";
    for (uint16_t ifed = 0; ifed < in_cabl.size(); ++ifed) {
      ss << in_cabl[ifed].first << " ";
    }
    LogTrace(mlRawToDigi_) << ss.str();
  }

  // Strip tracker FEDs
  if (edm::isDebugEnabled()) {
    stringstream ss;
    ss << "[SiStripFEDRawDataAnalyzer::" << __func__ << "]"
       << " Found " << trk_feds.size() << " strip tracker FEDs";
    edm::LogVerbatim(mlRawToDigi_) << ss.str();
    ss << " with ids (and buffer sizes [bytes]): ";
    for (uint16_t ifed = 0; ifed < trk_feds.size(); ++ifed) {
      ss << trk_feds[ifed].first << "(" << trk_feds[ifed].second << ") ";
    }
    LogTrace(mlRawToDigi_) << ss.str();
  }

  // Containers storing fed ids and channels
  ChannelsMap construct;  // errors on contruction of object
  ChannelsMap channel_construct;
  ChannelsMap channels_with_data;
  ChannelsMap channels_missing_data;
  ChannelsMap cabling_missing_channels;
  ChannelsMap connected;

  // Analyze strip tracker FED buffers in data
  for (uint16_t ii = 0; ii < trk_feds.size(); ++ii) {
    uint16_t ifed = trk_feds[ii].first;

    // record connections
    auto channels = cabling->fedConnections(ifed);
    for (uint16_t chan = 0; chan < channels.size(); ++chan) {
      if (channels[chan].isConnected()) {
        connected[ifed].push_back(channels[chan].fedCh());
      }
    }

    const FEDRawData& fed = buffers->FEDData(static_cast<int>(ifed));

    // construct buffer
    std::unique_ptr<FEDBuffer> buffer;
    if (FEDBufferStatusCode::SUCCESS != preconstructCheckFEDBuffer(fed)) {
      construct[ifed].push_back(0);
    } else {
      buffer = std::make_unique<FEDBuffer>(fed);
      if (FEDBufferStatusCode::SUCCESS != buffer->findChannels()) {
        construct[ifed].push_back(0);
      }
    }

    // readout mode
    sistrip::FEDReadoutMode mode = buffer->readoutMode();

    // loop channels
    for (uint16_t ichan = 0; ichan < 96; ++ichan) {
      // check if channel is good

      if (!buffer->channelGood(ichan, true)) {
        channel_construct[ifed].push_back(ichan);
        continue;
      }

      // find channel data

      if (mode == sistrip::READOUT_MODE_ZERO_SUPPRESSED && buffer->channel(ichan).length() > 7)
        channels_with_data[ifed].push_back(ichan);

      else if (mode == sistrip::READOUT_MODE_ZERO_SUPPRESSED_LITE10 && buffer->channel(ichan).length() > 2)
        channels_with_data[ifed].push_back(ichan);

      else if (mode == sistrip::READOUT_MODE_VIRGIN_RAW && buffer->channel(ichan).length() > 3)
        channels_with_data[ifed].push_back(ichan);

      else if (mode == sistrip::READOUT_MODE_PROC_RAW && buffer->channel(ichan).length() > 3)
        channels_with_data[ifed].push_back(ichan);

      else if (mode == sistrip::READOUT_MODE_SCOPE && buffer->channel(ichan).length() > 3)
        channels_with_data[ifed].push_back(ichan);

      else
        channels_missing_data[ifed].push_back(ichan);
    }

    // check if channels with data are in cabling
    uint16_t chan = 0;
    bool found = true;
    while (chan < connected[ifed].size() && found) {
      Channels::iterator iter = channels_with_data[ifed].begin();
      while (iter < channels_with_data[ifed].end()) {
        if (*iter == connected[ifed][chan]) {
          break;
        }
        iter++;
      }
      found = iter != channels_with_data[ifed].end();
      if (!found) {
        cabling_missing_channels[ifed].push_back(connected[ifed][chan]);
      }
      chan++;
    }
  }  // fed loop

  // constructing SiStripFEDBuffers
  if (edm::isDebugEnabled()) {
    stringstream ss;
    ss << "[SiStripFEDRawDataAnalyzer::" << __func__ << "]"
       << " Found " << construct.size() << " FED buffers that fail on construction of a FEDBuffer";
    edm::LogVerbatim(mlRawToDigi_) << ss.str();
    ss << " with ids: ";
    for (ChannelsMap::iterator iter = construct.begin(); iter != construct.end(); ++iter) {
      ss << iter->first << " ";
    }
    LogTrace(mlRawToDigi_) << ss.str();
  }

  // channel check
  if (edm::isDebugEnabled()) {
    stringstream ss;
    ss << "[SiStripFEDRawDataAnalyzer::" << __func__ << "]"
       << " Found " << channel_construct.size() << " FED channels that fail on FEDBuffer::channelGood()";
    edm::LogVerbatim(mlRawToDigi_) << ss.str();
    ss << " with ids: ";
    for (ChannelsMap::iterator iter = channel_construct.begin(); iter != channel_construct.end(); ++iter) {
      ss << iter->first << " ";
    }
    LogTrace(mlRawToDigi_) << ss.str();
  }

  //  connected channels with data
  if (edm::isDebugEnabled()) {
    stringstream ss;
    ss << "[SiStripFEDRawDataAnalyzer::" << __func__ << "]"
       << " Found " << channels_with_data.size() << " FED buffers with data in connected channels";
    edm::LogVerbatim(mlRawToDigi_) << ss.str();
    ss << " with ids (and number of connected channels with data): ";
    for (ChannelsMap::iterator iter = channels_with_data.begin(); iter != channels_with_data.end(); ++iter) {
      ss << iter->first << "(" << iter->second.size() << ") ";
    }
    LogTrace(mlRawToDigi_) << ss.str();
  }

  //  connected channels with missing data
  if (edm::isDebugEnabled()) {
    stringstream ss;
    ss << "[SiStripFEDRawDataAnalyzer::" << __func__ << "]"
       << " Found " << channels_missing_data.size() << " FED buffers with connected channels and missing data";
    edm::LogVerbatim(mlRawToDigi_) << ss.str();
    ss << " with ids (and number of connected channels with missing data): ";
    for (ChannelsMap::iterator iter = channels_missing_data.begin(); iter != channels_missing_data.end(); ++iter) {
      ss << iter->first << "(" << iter->second.size() << ") ";
    }
    LogTrace(mlRawToDigi_) << ss.str();
  }

  //  channels with data but not in cabling
  if (edm::isDebugEnabled()) {
    stringstream ss;
    ss << "[SiStripFEDRawDataAnalyzer::" << __func__ << "]"
       << " Found " << cabling_missing_channels.size()
       << " FED buffers with data in channels that are missing from cabling";
    edm::LogVerbatim(mlRawToDigi_) << ss.str();
    ss << " with ids (number of channels containing data that are missing from cabling, channel number): ";
    for (ChannelsMap::iterator iter = cabling_missing_channels.begin(); iter != cabling_missing_channels.end();
         ++iter) {
      ss << iter->first << "(" << iter->second.size() << ":";
      for (Channels::iterator jter = iter->second.begin(); jter != iter->second.end(); ++jter) {
        ss << *jter << ",";
      }
      ss << ") ";
    }
    LogTrace(mlRawToDigi_) << ss.str();
  }
}
