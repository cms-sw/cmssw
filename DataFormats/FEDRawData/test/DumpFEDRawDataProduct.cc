/** \file
 * 
 * 
 * \author N. Amapane - S. Argiro'
 *
*/

#include <iostream>
#include <iomanip>

#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;

namespace {
  template <typename T>
  std::set<T> make_set(std::vector<T> const& v) {
    std::set<T> s;
    for (auto const& e : v)
      s.insert(e);
    return s;
  }

  template <typename T>
  std::set<T> make_set(std::vector<T>&& v) {
    std::set<T> s;
    for (auto& e : v)
      s.insert(std::move(e));
    return s;
  }
}  // anonymous namespace

namespace test {

  class DumpFEDRawDataProduct : public edm::global::EDAnalyzer<> {
  private:
    const std::set<int> feds_;
    const edm::EDGetTokenT<FEDRawDataCollection> token_;
    const bool dumpPayload_;

  public:
    DumpFEDRawDataProduct(const ParameterSet& pset)
        : feds_(make_set(pset.getUntrackedParameter<std::vector<int>>("feds"))),
          token_(consumes<FEDRawDataCollection>(pset.getUntrackedParameter<edm::InputTag>("label"))),
          dumpPayload_(pset.getUntrackedParameter<bool>("dumpPayload")) {}

    void analyze(edm::StreamID sid, const Event& e, const EventSetup& c) const override {
      edm::LogSystem out("DumpFEDRawDataProduct");
      Handle<FEDRawDataCollection> rawdata;
      e.getByToken(token_, rawdata);
      for (int i = 0; i <= FEDNumbering::lastFEDId(); i++) {
        const FEDRawData& data = rawdata->FEDData(i);
        size_t size = data.size();

        if (size > 0 && (feds_.empty() || feds_.find(i) != feds_.end())) {
          out << "FED# " << std::setw(4) << i << " " << std::setw(8) << size << " bytes ";

          FEDHeader header(data.data());
          FEDTrailer trailer(data.data() + size - FEDTrailer::length);

          out << " L1Id: " << std::setw(8) << header.lvl1ID();
          out << " BXId: " << std::setw(4) << header.bxID();
          out << '\n';

          if (dumpPayload_) {
            const uint64_t* payload = (uint64_t*)(data.data());
            out << std::hex << std::setfill('0');
            for (unsigned int i = 0; i < data.size() / sizeof(uint64_t); i++) {
              out << std::setw(4) << i << "  " << std::setw(16) << payload[i] << '\n';
            }
            out << std::dec << std::setfill(' ');
          }

          if (not trailer.check()) {
            out << "    FED trailer check failed\n";
          }
          if (trailer.fragmentLength() * 8 != data.size()) {
            out << "    FED fragment size mismatch: " << trailer.fragmentLength() << " (fragment length) vs "
                << (float)data.size() / 8 << " (data size) words\n";
          }
        } else if (size == 0 && feds_.find(i) != feds_.end()) {
          out << "FED# " << std::setw(4) << i << " " << std::setw(8) << size << " bytes\n";
        }
      }
    }
  };

  DEFINE_FWK_MODULE(DumpFEDRawDataProduct);
}  // namespace test
