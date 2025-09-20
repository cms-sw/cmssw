/** \file
 * 
 * 
 * \author N. Amapane - S. Argiro'
 *
*/

#include <iostream>
#include <iomanip>

#include "DataFormats/FEDRawData/interface/RawDataBuffer.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

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
  public:
    DumpFEDRawDataProduct(const ParameterSet& pset)
        : feds_(make_set(pset.getUntrackedParameter<std::vector<int>>("feds"))),
          phase1_token_(consumes<FEDRawDataCollection>(pset.getUntrackedParameter<edm::InputTag>("label"))),
          phase2_token_(consumes<RawDataBuffer>(pset.getUntrackedParameter<edm::InputTag>("label"))),
          dumpPayload_(pset.getUntrackedParameter<bool>("dumpPayload")),
          usePhase2_(pset.getUntrackedParameter<bool>("usePhase2")) {}

    static void fillDescriptions(ConfigurationDescriptions& descriptions) {
      ParameterSetDescription desc;
      desc.addUntracked<std::vector<int>>("feds")->setComment("List of FED IDs of interest");
      desc.addUntracked<edm::InputTag>("label")->setComment("Label for the raw data collection");
      desc.addUntracked<bool>("dumpPayload")->setComment("Enable payload dump");
      desc.addUntracked<bool>("usePhase2")
          ->setComment("Use Phase 2 RawDataBuffer instead of Phase 1's FEDRawDataCollection");
      descriptions.add("dumpFEDdata", desc);
    }

    void analyze(edm::StreamID sid, const Event& e, const EventSetup& c) const override {
      if (usePhase2_)
        analyzePhase2(e, c);
      else
        analyzePhase1(e, c);
    }

  private:
    const std::set<int> feds_;
    const edm::EDGetTokenT<FEDRawDataCollection> phase1_token_;
    const edm::EDGetTokenT<RawDataBuffer> phase2_token_;
    const bool dumpPayload_, usePhase2_;

    void analyzePhase1(const Event& e, const EventSetup& c) const {
      edm::LogSystem out("DumpFEDRawDataProduct");

      Handle<FEDRawDataCollection> rawdata;
      e.getByToken(phase1_token_, rawdata);
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

    void analyzePhase2(const Event& e, const EventSetup& c) const {
      edm::LogSystem out("DumpFEDRawDataProduct");

      Handle<RawDataBuffer> rawdata;
      e.getByToken(phase2_token_, rawdata);
      for (const auto& it : rawdata->map()) {
        auto fedid = it.first;
        if (!feds_.empty() && feds_.find(fedid) == feds_.end())
          continue;

        auto offset = it.second.first;
        auto size = it.second.second;
        out << "FED# " << std::setw(4) << fedid << " " << std::setw(8) << size << " bytes  offset=" << std::setw(8)
            << offset;

        const auto& data = rawdata->fragmentData(fedid);
        const auto* start_fed_data = &(data.data().front());
        FEDHeader header(start_fed_data);
        FEDTrailer trailer(start_fed_data + size - FEDTrailer::length);

        out << " L1Id: " << std::setw(8) << header.lvl1ID();
        out << " BXId: " << std::setw(4) << header.bxID();
        out << '\n';

        if (dumpPayload_) {
          const uint64_t* payload = (uint64_t*)(start_fed_data);
          out << std::hex << std::setfill('0');
          for (unsigned int i = 0; i < size / sizeof(uint64_t); i++) {
            out << std::setw(4) << i << "  " << std::setw(16) << payload[i] << '\n';
          }
          out << std::dec << std::setfill(' ');
        }

        if (not trailer.check()) {
          out << "    FED trailer check failed\n";
        }
        if (trailer.fragmentLength() * 8 != size) {
          out << "    FED fragment size mismatch: " << trailer.fragmentLength() << " (fragment length) vs " << size / 8
              << " (data size) words\n";
        }
      }
    }
  };

  DEFINE_FWK_MODULE(DumpFEDRawDataProduct);
}  // namespace test
