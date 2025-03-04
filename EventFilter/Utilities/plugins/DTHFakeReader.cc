#include "DTHFakeReader.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/TCDS/interface/TCDSRaw.h"

//#include "EventFilter/Utilities/interface/GlobalEventNumber.h"
#include "EventFilter/Utilities/interface/crc32c.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CLHEP/Random/RandGauss.h"

#include <cmath>
#include <sys/time.h>
#include <cstring>
#include <cstdlib>
#include <chrono>

//using namespace edm;
namespace evf {

  constexpr unsigned minOrbitBx = 1;
  constexpr unsigned maxOrbitBx = 2464;
  constexpr unsigned avgEventsPerOrbit = 70;

  //constexpr unsigned h_size_ = 8;//for SLink FEDs
  //constexpr unsigned t_size_ = 8;

  constexpr unsigned h_size_ = sizeof(SLinkRocketHeader_v3);
  constexpr unsigned t_size_ = sizeof(SLinkRocketTrailer_v3);

  constexpr double rndFactor = (maxOrbitBx - minOrbitBx + 1) / (double(avgEventsPerOrbit) * RAND_MAX);

  DTHFakeReader::DTHFakeReader(const edm::ParameterSet& pset)
      : fillRandom_(pset.getUntrackedParameter<bool>("fillRandom", false)),
        meansize_(pset.getUntrackedParameter<unsigned int>("meanSize", 1024)),
        width_(pset.getUntrackedParameter<unsigned int>("width", 1024)),
        injected_errors_per_million_events_(pset.getUntrackedParameter<unsigned int>("injectErrPpm", 0)),
        sourceIdList_(
            pset.getUntrackedParameter<std::vector<unsigned int>>("sourceIdList", std::vector<unsigned int>())),
        modulo_error_events_(injected_errors_per_million_events_ ? 1000000 / injected_errors_per_million_events_
                                                                 : 0xffffffff) {
    if (fillRandom_) {
      //intialize random seed
      auto time_count =
          static_cast<long unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
      std::srand(time_count & 0xffffffff);
    }
    produces<FEDRawDataCollection>();
  }

  void DTHFakeReader::fillRawData(edm::Event& e, FEDRawDataCollection*& data) {
    // a null pointer is passed, need to allocate the fed collection (reusing it as container)
    data = new FEDRawDataCollection();
    //auto ls = e.luminosityBlock();
    //this will be used as orbit counter
    edm::EventNumber_t orbitId = e.id().event();

    //generate eventID. Orbits start from 0 or 1?
    std::vector<uint64_t> eventIdList_;
    std::map<unsigned int, std::map<uint64_t, uint32_t>> randFedSizes;
    for (auto sourceId : sourceIdList_) {
      randFedSizes[sourceId] = std::map<uint64_t, uint32_t>();
    }

    //randomize which orbit was accepted
    for (unsigned i = minOrbitBx; i <= maxOrbitBx; i++) {
      if ((std::rand() * rndFactor) < 1) {
        uint64_t eventId = orbitId * maxOrbitBx + i;
        eventIdList_.push_back(eventId);
        for (auto sourceId : sourceIdList_) {
          float logsiz = CLHEP::RandGauss::shoot(std::log(meansize_), std::log(meansize_) - std::log(width_ / 2.));
          size_t size = int(std::exp(logsiz));
          size -= size % 16;  // all blocks aligned to 128 bit words (with header+trailer being 16, this remains valid)
          if (!size)
            size = 16;
          randFedSizes[sourceId][eventId] = size;
        }
      }
    }

    for (auto sourceId : sourceIdList_) {
      FEDRawData& feddata = data->FEDData(sourceId);

      auto size = sizeof(DTHOrbitHeader_v1);
      for (auto eventId : eventIdList_)
        size += randFedSizes[sourceId][eventId] + h_size_ + t_size_ + sizeof(DTHFragmentTrailer_v1);
      feddata.resize(size);

      uint64_t fragments_size_bytes = sizeof(DTHOrbitHeader_v1);
      //uint32_t runningChecksum = 0xffffffffU;
      uint32_t runningChecksum = 0;
      for (auto eventId : eventIdList_) {
        unsigned char* fedaddr = feddata.data() + fragments_size_bytes;
        //fragments_size_bytes += fillFED(fedaddr, sourceId, eventId, randFedSizes[sourceId][eventId], runningChecksum);
        fragments_size_bytes +=
            fillSLRFED(fedaddr, sourceId, eventId, orbitId, randFedSizes[sourceId][eventId], runningChecksum);
      }
      //in place construction
      new (feddata.data()) DTHOrbitHeader_v1(sourceId,
                                             e.id().run(),
                                             orbitId,
                                             eventIdList_.size(),
                                             fragments_size_bytes >> evf::DTH_WORD_NUM_BYTES_SHIFT,
                                             0,
                                             runningChecksum);
    }
  }

  void DTHFakeReader::produce(edm::Event& e, edm::EventSetup const& es) {
    edm::Handle<FEDRawDataCollection> rawdata;
    FEDRawDataCollection* fedcoll = nullptr;
    fillRawData(e, fedcoll);
    std::unique_ptr<FEDRawDataCollection> bare_product(fedcoll);
    e.put(std::move(bare_product));
  }

  uint32_t DTHFakeReader::fillSLRFED(unsigned char* buf,
                                     const uint32_t sourceId,
                                     edm::EventNumber_t eventId,
                                     const uint32_t orbitId,
                                     uint32_t size,
                                     uint32_t& accum_crc32c) {
    // Generate size...
    const unsigned h_size_ = sizeof(SLinkRocketHeader_v3);
    const unsigned t_size_ = sizeof(SLinkRocketTrailer_v3);

    uint32_t totsize = size + h_size_ + t_size_ + sizeof(DTHFragmentTrailer_v1);
    const unsigned fragsize = size + h_size_ + t_size_;

    //Fill SLinkRocket header
    uint8_t emu_status = 2;  //set 2 indicating fragment generated by DTH (emulator)
    uint16_t l1a_types = 1;  //set provisionally to 1, to be revised later
    uint8_t l1a_phys = 0;
    new ((void*)buf) SLinkRocketHeader_v3(sourceId, l1a_types, l1a_phys, emu_status, eventId);

    // Payload = all 0s or random
    if (fillRandom_) {
      //fill FED with random values
      size_t size_ui = size - size % sizeof(unsigned int);
      for (size_t i = 0; i < size_ui; i += sizeof(unsigned int)) {
        *((unsigned int*)(buf + h_size_ + i)) = (unsigned int)std::rand();
      }
      //remainder
      for (size_t i = size_ui; i < size; i++) {
        *(buf + h_size_ + i) = std::rand() & 0xff;
      }
    }

    //Fill SLinkRocket trailer
    uint16_t crc = 0;  // FIXME : get CRC16
    uint16_t bxid = 0;
    uint8_t status = 0;
    //size is in bytes, it will be converted by constructor
    new ((void*)(buf + h_size_ + size))
        SLinkRocketTrailer_v3(status, crc, orbitId, bxid, fragsize >> evf::SLR_WORD_NUM_BYTES_SHIFT, crc);

    //fill DTH fragment trailer
    void* dthTrailerAddr = buf + fragsize;
    new (dthTrailerAddr) DTHFragmentTrailer_v1(0, fragsize >> evf::DTH_WORD_NUM_BYTES_SHIFT, eventId, crc);

    //accumulate crc32 checksum
    accum_crc32c = crc32c(accum_crc32c, (const uint8_t*)buf, totsize);

    return totsize;
  }

  uint32_t DTHFakeReader::fillFED(
      unsigned char* buf, const int sourceId, edm::EventNumber_t eventId, uint32_t size, uint32_t& accum_crc32c) {
    // Generate size...
    const unsigned h_size = 8;
    const unsigned t_size = 8;

    //header+trailer+payload
    uint32_t totsize = size + h_size + t_size + sizeof(DTHFragmentTrailer_v1);

    // Generate header
    //FEDHeader::set(feddata.data(),
    FEDHeader::set(buf,
                   1,          // Trigger type
                   eventId,    // LV1_id (24 bits)
                   0,          // BX_id
                   sourceId);  // source_id

    // Payload = all 0s or random
    if (fillRandom_) {
      //fill FED with random values
      size_t size_ui = size - size % sizeof(unsigned int);
      for (size_t i = 0; i < size_ui; i += sizeof(unsigned int)) {
        *((unsigned int*)(buf + h_size + i)) = (unsigned int)std::rand();
      }
      //remainder
      for (size_t i = size_ui; i < size; i++) {
        *(buf + h_size + i) = std::rand() & 0xff;
      }
    }

    // Generate trailer
    int crc = 0;  // FIXME : get CRC16
    FEDTrailer::set(buf + h_size + size,
                    size / 8 + 2,  // in 64 bit words
                    crc,
                    0,   // Evt_stat
                    0);  // TTS bits

    //FIXME: accumulate crc32 checksum
    //crc32c = 0;

    void* dthTrailerAddr = buf + h_size + t_size + size;
    new (dthTrailerAddr)
        DTHFragmentTrailer_v1(0, (h_size + t_size + size) >> evf::DTH_WORD_NUM_BYTES_SHIFT, crc, eventId);
    return totsize;
  }

  void DTHFakeReader::beginLuminosityBlock(edm::LuminosityBlock const& iL, edm::EventSetup const& iE) {
    std::cout << "DTHFakeReader begin Lumi " << iL.luminosityBlock() << std::endl;
    fakeLs_ = iL.luminosityBlock();
  }

  void DTHFakeReader::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.setComment("Injector of generated DTH raw orbit fragments for DSAQ testing");
    desc.addUntracked<bool>("fillRandom", false);
    desc.addUntracked<unsigned int>("meanSize", 1024);
    desc.addUntracked<unsigned int>("width", 1024);
    desc.addUntracked<unsigned int>("injectErrPpm", 1024);
    desc.addUntracked<std::vector<unsigned int>>("sourceIdList", std::vector<unsigned int>());
    descriptions.add("DTHFakeReader", desc);
  }
}  //namespace evf
