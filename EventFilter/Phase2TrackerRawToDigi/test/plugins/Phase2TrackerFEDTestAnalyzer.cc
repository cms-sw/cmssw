// system includes
#include <utility>
#include <vector>

// user includes
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDBuffer.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDChannel.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDHeader.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDRawChannelUnpacker.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDZSChannelUnpacker.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/utils.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#define LOGPRINT edm::LogPrint("Phase2TrackerFEDTestAnalyzer")

/**
   @class Phase2TrackerFEDTestAnalyzer 
   @brief Analyzes contents of FED_test_ collection
*/

class Phase2TrackerFEDTestAnalyzer : public edm::one::EDAnalyzer<> {
public:
  typedef std::pair<uint16_t, uint16_t> Fed;
  typedef std::vector<Fed> Feds;
  typedef std::vector<uint16_t> Channels;
  typedef std::map<uint16_t, Channels> ChannelsMap;

  Phase2TrackerFEDTestAnalyzer(const edm::ParameterSet&);
  ~Phase2TrackerFEDTestAnalyzer();

  void beginJob();
  void analyze(const edm::Event&, const edm::EventSetup&);
  void endJob();

private:
  const edm::EDGetTokenT<FEDRawDataCollection> token_;
};

using namespace Phase2Tracker;
using namespace std;

// -----------------------------------------------------------------------------
//
Phase2TrackerFEDTestAnalyzer::Phase2TrackerFEDTestAnalyzer(const edm::ParameterSet& pset)
    : token_(consumes<FEDRawDataCollection>(pset.getParameter<edm::InputTag>("ProductLabel"))) {
  LogDebug("Phase2TrackerFEDTestAnalyzer") << "[Phase2TrackerFEDTestAnalyzer::" << __func__ << "]"
                                           << "Constructing object...";
}

// -----------------------------------------------------------------------------
//
Phase2TrackerFEDTestAnalyzer::~Phase2TrackerFEDTestAnalyzer() {
  LogDebug("Phase2TrackerFEDTestAnalyzer") << "[Phase2TrackerFEDTestAnalyzer::" << __func__ << "]"
                                           << " Destructing object...";
}

// -----------------------------------------------------------------------------
//
void Phase2TrackerFEDTestAnalyzer::beginJob() {
  LogDebug("Phase2TrackerFEDTestAnalyzer") << "[Phase2TrackerFEDTestAnalyzer::" << __func__ << "]";
}

// -----------------------------------------------------------------------------
//
void Phase2TrackerFEDTestAnalyzer::endJob() {
  LogDebug("Phase2TrackerFEDTestAnalyzer") << "[Phase2TrackerFEDTestAnalyzer::" << __func__ << "]";
}

// -----------------------------------------------------------------------------
//
void Phase2TrackerFEDTestAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  // Retrieve FEDRawData collection
  edm::Handle<FEDRawDataCollection> buffers;
  event.getByToken(token_, buffers);

  // Analyze strip tracker FED buffers in data
  size_t fedIndex;
  for (fedIndex = 0; fedIndex < Phase2Tracker::CMS_FED_ID_MAX; ++fedIndex) {
    const FEDRawData& fed = buffers->FEDData(fedIndex);
    if (fed.size() != 0 && fedIndex >= Phase2Tracker::FED_ID_MIN && fedIndex <= Phase2Tracker::FED_ID_MAX) {
      // construct buffer
      Phase2Tracker::Phase2TrackerFEDBuffer* buffer = 0;
      buffer = new Phase2Tracker::Phase2TrackerFEDBuffer(fed.data(), fed.size());

      LOGPRINT << " -------------------------------------------- ";
      LOGPRINT << " buffer debug ------------------------------- ";
      LOGPRINT << " -------------------------------------------- ";
      LOGPRINT << " buffer size : " << buffer->bufferSize();
      LOGPRINT << " fed id      : " << fedIndex;
      LOGPRINT << " -------------------------------------------- ";
      LOGPRINT << " tracker header debug ------------------------";
      LOGPRINT << " -------------------------------------------- ";

      Phase2TrackerFEDHeader tr_header = buffer->trackerHeader();
      LOGPRINT << " Version  : " << hex << setw(2) << (int)tr_header.getDataFormatVersion();
      LOGPRINT << " Mode     : " << hex << setw(2) << (int)tr_header.getDebugMode();
      LOGPRINT << " Type     : " << hex << setw(2) << (int)tr_header.getEventType();
      LOGPRINT << " Readout  : " << hex << setw(2) << (int)tr_header.getReadoutMode();
      LOGPRINT << " Status   : " << hex << setw(16) << (int)tr_header.getGlibStatusCode();
      LOGPRINT << " FE stat  : ";
      for (int i = 15; i >= 0; i--) {
        if ((tr_header.frontendStatus())[i]) {
          LOGPRINT << "1";
        } else {
          LOGPRINT << "0";
        }
      }
      LOGPRINT << "\n";
      LOGPRINT << " Nr CBC   : " << hex << setw(16) << (int)tr_header.getNumberOfCBC();
      LOGPRINT << " CBC stat : ";
      for (int i = 0; i < tr_header.getNumberOfCBC(); i++) {
        LOGPRINT << hex << setw(2) << (int)tr_header.CBCStatus()[i] << " ";
      }
      LOGPRINT << "\n";
      LOGPRINT << " -------------------------------------------- ";
      LOGPRINT << " Payload  ----------------------------------- ";
      LOGPRINT << " -------------------------------------------- ";

      // loop channels
      int ichan = 0;
      for (int ife = 0; ife < 16; ife++) {
        for (int icbc = 0; icbc < 16; icbc++) {
          const Phase2TrackerFEDChannel& channel = buffer->channel(ichan);
          if (channel.length() > 0) {
            LOGPRINT << dec << " reading channel : " << icbc << " on FE " << ife;
            LOGPRINT << dec << " with length  : " << (int)channel.length();
            Phase2TrackerFEDRawChannelUnpacker unpacker = Phase2TrackerFEDRawChannelUnpacker(channel);
            while (unpacker.hasData()) {
              LOGPRINT << (unpacker.stripOn() ? "1" : "_");
              unpacker++;
            }
            LOGPRINT << "\n";
          }
          ichan++;
        }
      }  // end loop on channels
    }
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(Phase2TrackerFEDTestAnalyzer);
