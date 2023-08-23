#include "CondFormats/DataRecord/interface/Phase2TrackerCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/Phase2TrackerCabling.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDBuffer.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDChannel.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDHeader.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDRawChannelUnpacker.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDZSChannelUnpacker.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/utils.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace Phase2Tracker {

  class Phase2TrackerDigiProducer : public edm::stream::EDProducer<> {
  public:
    /// constructor
    Phase2TrackerDigiProducer(const edm::ParameterSet& pset);
    /// default constructor
    ~Phase2TrackerDigiProducer() override = default;
    void beginRun(edm::Run const&, edm::EventSetup const&) override;
    void produce(edm::Event&, const edm::EventSetup&) override;

  private:
    const edm::ESGetToken<Phase2TrackerCabling, Phase2TrackerCablingRcd> ph2CablingESToken_;

    unsigned int runNumber_;
    edm::EDGetTokenT<FEDRawDataCollection> token_;
    const Phase2TrackerCabling* cabling_;
    uint32_t cacheId_;
    DetIdCollection detids_;
    class Registry {
    public:
      /// constructor
      Registry(uint32_t aDetid, uint16_t firstStrip, size_t indexInVector, uint16_t numberOfDigis)
          : detid(aDetid), first(firstStrip), index(indexInVector), length(numberOfDigis) {}
      /// < operator to sort registries
      bool operator<(const Registry& other) const {
        return (detid != other.detid ? detid < other.detid : first < other.first);
      }
      /// public data members
      uint32_t detid;
      uint16_t first;
      size_t index;
      uint16_t length;
    };
    std::vector<Registry> proc_work_registry_;
    std::vector<Phase2TrackerDigi> proc_work_digis_;
  };
}  // namespace Phase2Tracker

#include "FWCore/Framework/interface/MakerMacros.h"
typedef Phase2Tracker::Phase2TrackerDigiProducer Phase2TrackerDigiProducer;
DEFINE_FWK_MODULE(Phase2TrackerDigiProducer);

using namespace std;

namespace Phase2Tracker {

  Phase2TrackerDigiProducer::Phase2TrackerDigiProducer(const edm::ParameterSet& pset)
      : ph2CablingESToken_(esConsumes<edm::Transition::BeginRun>()), runNumber_(0), cabling_(nullptr), cacheId_(0) {
    // define product
    produces<edm::DetSetVector<Phase2TrackerDigi>>("ProcessedRaw");
    token_ = consumes<FEDRawDataCollection>(pset.getParameter<edm::InputTag>("ProductLabel"));
  }

  void Phase2TrackerDigiProducer::beginRun(edm::Run const& run, edm::EventSetup const& es) {
    // fetch cabling from event setup
    cabling_ = &es.getData(ph2CablingESToken_);
  }

  void Phase2TrackerDigiProducer::produce(edm::Event& event, const edm::EventSetup& es) {
    // empty vectors for the next event
    proc_work_registry_.clear();
    proc_work_digis_.clear();

    // Retrieve FEDRawData collection
    edm::Handle<FEDRawDataCollection> buffers;
    event.getByToken(token_, buffers);

    // Analyze strip tracker FED buffers in data
    size_t fedIndex;
    for (fedIndex = Phase2Tracker::FED_ID_MIN; fedIndex <= Phase2Tracker::CMS_FED_ID_MAX; ++fedIndex) {
      const FEDRawData& fed = buffers->FEDData(fedIndex);
      if (fed.size() != 0) {
        // construct buffer
        Phase2Tracker::Phase2TrackerFEDBuffer* buffer = nullptr;
        buffer = new Phase2Tracker::Phase2TrackerFEDBuffer(fed.data(), fed.size());

#ifdef EDM_ML_DEBUG
        std::ostringstream ss;
        ss << " -------------------------------------------- " << endl;
        ss << " buffer debug ------------------------------- " << endl;
        ss << " -------------------------------------------- " << endl;
        ss << " buffer size : " << buffer->bufferSize() << endl;
        ss << " fed id      : " << fedIndex << endl;
        ss << " -------------------------------------------- " << endl;
        ss << " tracker header debug ------------------------" << endl;
        ss << " -------------------------------------------- " << endl;
        LogTrace("Phase2TrackerDigiProducer") << ss.str();
        ss.clear();
        ss.str("");

        Phase2TrackerFEDHeader tr_header = buffer->trackerHeader();
        ss << " Version  : " << hex << setw(2) << (int)tr_header.getDataFormatVersion() << endl;
        ss << " Mode     : " << hex << setw(2) << tr_header.getDebugMode() << endl;
        ss << " Type     : " << hex << setw(2) << (int)tr_header.getEventType() << endl;
        ss << " Readout  : " << hex << setw(2) << tr_header.getReadoutMode() << endl;
        ss << " Condition Data : " << (tr_header.getConditionData() ? "Present" : "Absent") << "\n";
        ss << " Data Type      : " << (tr_header.getDataType() ? "Real" : "Fake") << "\n";
        ss << " Status   : " << hex << setw(16) << (int)tr_header.getGlibStatusCode() << endl;
        ss << " FE stat  : ";
        for (int i = 15; i >= 0; i--) {
          if ((tr_header.frontendStatus())[i]) {
            ss << "1";
          } else {
            ss << "0";
          }
        }
        ss << endl;
        ss << " Nr CBC   : " << hex << setw(16) << (int)tr_header.getNumberOfCBC() << endl;
        ss << " CBC stat : ";
        for (int i = 0; i < tr_header.getNumberOfCBC(); i++) {
          ss << hex << setw(2) << (int)tr_header.CBCStatus()[i] << " ";
        }
        ss << endl;
        LogTrace("Phase2TrackerDigiProducer") << ss.str();
        ss.clear();
        ss.str("");
        ss << " -------------------------------------------- " << endl;
        ss << " Payload  ----------------------------------- " << endl;
        ss << " -------------------------------------------- " << endl;
#endif

        // loop channels
        int ichan = 0;
        for (int ife = 0; ife < MAX_FE_PER_FED; ife++) {
          for (int icbc = 0; icbc < MAX_CBC_PER_FE; icbc++) {
            const Phase2TrackerFEDChannel& channel = buffer->channel(ichan);
            if (channel.length() > 0) {
              // get fedid from cabling
              const Phase2TrackerModule mod = cabling_->findFedCh(std::make_pair(fedIndex, ife));
              uint32_t detid = mod.getDetid();
#ifdef EDM_ML_DEBUG
              ss << dec << " id from cabling : " << detid << endl;
              ss << dec << " reading channel : " << icbc << " on FE " << ife;
              ss << dec << " with length  : " << (int)channel.length() << endl;
#endif

              // container for this channel's digis
              std::vector<Phase2TrackerDigi> stripsTop;
              std::vector<Phase2TrackerDigi> stripsBottom;

              // unpacking data
              Phase2TrackerFEDRawChannelUnpacker unpacker = Phase2TrackerFEDRawChannelUnpacker(channel);
              while (unpacker.hasData()) {
                if (unpacker.stripOn()) {
                  if (unpacker.stripIndex() % 2) {
                    stripsTop.push_back(Phase2TrackerDigi((int)(STRIPS_PER_CBC * icbc + unpacker.stripIndex()) / 2, 0));
#ifdef EDM_ML_DEBUG
                    ss << "t";
#endif
                  } else {
                    stripsBottom.push_back(
                        Phase2TrackerDigi((int)(STRIPS_PER_CBC * icbc + unpacker.stripIndex()) / 2, 0));
#ifdef EDM_ML_DEBUG
                    ss << "b";
#endif
                  }
                } else {
#ifdef EDM_ML_DEBUG
                  ss << "_";
#endif
                }
                unpacker++;
              }
#ifdef EDM_ML_DEBUG
              ss << endl;
              LogTrace("Phase2TrackerDigiProducer") << ss.str();
              ss.clear();
              ss.str("");
#endif

              // store beginning and end of this digis for this detid and add this registry to the list
              // and store data
              Registry regItemTop(detid + 1, STRIPS_PER_CBC * icbc / 2, proc_work_digis_.size(), stripsTop.size());
              proc_work_registry_.push_back(regItemTop);
              proc_work_digis_.insert(proc_work_digis_.end(), stripsTop.begin(), stripsTop.end());
              Registry regItemBottom(
                  detid + 2, STRIPS_PER_CBC * icbc / 2, proc_work_digis_.size(), stripsBottom.size());
              proc_work_registry_.push_back(regItemBottom);
              proc_work_digis_.insert(proc_work_digis_.end(), stripsBottom.begin(), stripsBottom.end());
            }
            ichan++;
          }
        }  // end loop on channels
        // store digis in edm collections
        std::sort(proc_work_registry_.begin(), proc_work_registry_.end());
        std::vector<edm::DetSet<Phase2TrackerDigi>> sorted_and_merged;

        edm::DetSetVector<Phase2TrackerDigi>* pr = new edm::DetSetVector<Phase2TrackerDigi>();

        std::vector<Registry>::iterator it = proc_work_registry_.begin(), it2 = it + 1, end = proc_work_registry_.end();
        while (it < end) {
          sorted_and_merged.push_back(edm::DetSet<Phase2TrackerDigi>(it->detid));
          std::vector<Phase2TrackerDigi>& digis = sorted_and_merged.back().data;
          // first count how many digis we have
          size_t len = it->length;
          for (it2 = it + 1; (it2 != end) && (it2->detid == it->detid); ++it2) {
            len += it2->length;
          }
          // reserve memory
          digis.reserve(len);
          // push them in
          for (it2 = it + 0; (it2 != end) && (it2->detid == it->detid); ++it2) {
            digis.insert(digis.end(), &proc_work_digis_[it2->index], &proc_work_digis_[it2->index + it2->length]);
          }
          it = it2;
        }

        edm::DetSetVector<Phase2TrackerDigi> proc_raw_dsv(sorted_and_merged, true);
        pr->swap(proc_raw_dsv);
        event.put(std::unique_ptr<edm::DetSetVector<Phase2TrackerDigi>>(pr), "ProcessedRaw");
        delete buffer;
      }
    }
  }
}  // namespace Phase2Tracker
