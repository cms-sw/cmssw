#ifndef EventFilter_RPCRawToDigi_RPCAMCRawToDigi_h
#define EventFilter_RPCRawToDigi_RPCAMCRawToDigi_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/CRC16.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/RPCDigi/interface/RPCAMCLinkCounters.h"
#include "EventFilter/RPCRawToDigi/interface/RPCAMC13Record.h"

namespace edm {
  class ConfigurationDescriptions;
  class Event;
  class EventSetup;
  class ParameterSet;
  class Run;
}  // namespace edm

class RPCAMCUnpacker;

class RPCAMCRawToDigi : public edm::stream::EDProducer<> {
public:
  RPCAMCRawToDigi(edm::ParameterSet const &config);
  ~RPCAMCRawToDigi() override;

  static void compute_crc16_64bit(std::uint16_t &crc, std::uint64_t const &word);

  static void fillDescriptions(edm::ConfigurationDescriptions &descs);

  void beginRun(edm::Run const &run, edm::EventSetup const &setup) override;
  void produce(edm::Event &event, edm::EventSetup const &setup) override;

protected:
  bool processCDFHeaders(int fed,
                         std::uint64_t const *&word,
                         std::uint64_t const *&word_end,
                         std::uint16_t &crc,
                         RPCAMCLinkCounters &counters) const;
  bool processCDFTrailers(int fed,
                          unsigned int nwords,
                          std::uint64_t const *&word,
                          std::uint64_t const *&word_end,
                          std::uint16_t &crc,
                          RPCAMCLinkCounters &counters) const;
  bool processBlocks(int fed,
                     std::uint64_t const *&word,
                     std::uint64_t const *word_end,
                     std::uint16_t &crc,
                     RPCAMCLinkCounters &counters,
                     std::map<RPCAMCLink, rpcamc13::AMCPayload> &amc_payload) const;

protected:
  edm::EDGetTokenT<FEDRawDataCollection> raw_token_;

  bool calculate_crc_, fill_counters_;

  std::unique_ptr<RPCAMCUnpacker> rpc_unpacker_;
};

inline void RPCAMCRawToDigi::compute_crc16_64bit(
    std::uint16_t &crc, std::uint64_t const &word) {  // overcome constness problem evf::compute_crc_64bit
  unsigned char const *uchars(reinterpret_cast<unsigned char const *>(&word));
  for (unsigned char const *uchar = uchars + 7; uchar >= uchars; --uchar) {
    crc = evf::compute_crc_8bit(crc, *uchar);
  }
}

#endif  // EventFilter_RPCRawToDigi_RPCAMCRawToDigi_h
