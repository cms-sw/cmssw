#include <string>
#include <iostream>
#include <vector>
#include <array>
#include <cstdint>

#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

namespace {
  std::ostream& operator<<(std::ostream& s, const HFQIE10Info& i) {
    s << i.id() << ": " << i.energy() << " GeV"
      << ", t= " << i.timeRising() << " to " << i.timeFalling() << " ns";
    return s;
  }

  std::ostream& operator<<(std::ostream& s, const HFPreRecHit& hit) {
    s << "{ ";
    const HFQIE10Info* i = hit.getHFQIE10Info(0);
    if (i) {
      s << *i;
    }
    s << " }, ";
    s << "{ ";
    i = hit.getHFQIE10Info(1);
    if (i) {
      s << *i;
    }
    s << " }";
    return s;
  }

  template <std::size_t N>
  void printBits(std::ostream& s, const std::array<uint32_t, N>& allbits, const std::vector<int>& bits) {
    const int maxbit = N * 32;
    const unsigned len = bits.size();
    for (unsigned i = 0; i < len; ++i) {
      const int bitnum = bits[i];
      if (bitnum >= 0 && bitnum < maxbit) {
        const unsigned ibit = bitnum % 32;
        const bool bit = (allbits[bitnum / 32] & (1U << ibit)) >> ibit;
        s << bit;
      } else
        s << '-';
    }
  }

  void printRecHitAuxInfo(std::ostream& s, const HFPreRecHit& i, const std::vector<int>& bits, bool) {}

  void printRecHitAuxInfo(std::ostream& s, const HBHERecHit& i, const std::vector<int>& bits, const bool plan1) {
    if (plan1 && i.isMerged()) {
      // This is a "Plan 1" combined rechit
      std::vector<HcalDetId> ids;
      i.getMergedIds(&ids);
      const unsigned n = ids.size();
      s << "; merged " << n << ": ";
      for (unsigned j = 0; j < n; ++j) {
        if (j)
          s << ", ";
        s << ids[j];
      }
    }
    if (!bits.empty()) {
      std::array<uint32_t, 4> allbits;
      allbits[0] = i.flags();
      allbits[1] = i.aux();
      allbits[2] = i.auxHBHE();
      allbits[3] = i.auxPhase1();
      s << "; bits: ";
      printBits(s, allbits, bits);
    }
  }

  void printRecHitAuxInfo(std::ostream& s, const HFRecHit& i, const std::vector<int>& bits, bool) {
    if (!bits.empty()) {
      std::array<uint32_t, 3> allbits;
      allbits[0] = i.flags();
      allbits[1] = i.aux();
      allbits[2] = i.getAuxHF();
      s << "; bits: ";
      printBits(s, allbits, bits);
    }
  }
}  // namespace

using namespace std;

class HcalRecHitDump : public edm::stream::EDAnalyzer<> {
public:
  explicit HcalRecHitDump(edm::ParameterSet const& conf);
  void analyze(edm::Event const& e, edm::EventSetup const& c) override;

private:
  string hbhePrefix_;
  string hfPrefix_;
  string hfprePrefix_;
  std::vector<int> bits_;
  bool printPlan1Info_;

  edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
  edm::EDGetTokenT<HFRecHitCollection> tok_hf_;
  edm::EDGetTokenT<HFPreRecHitCollection> tok_prehf_;

  unsigned long long counter_;

  template <class Collection, class Token>
  void analyzeT(edm::Event const& e,
                const Token& tok,
                const char* name,
                const string& prefix,
                const bool printPlan1Info = false) const {
    cout << prefix << " rechit dump " << counter_ << endl;

    edm::Handle<Collection> coll;
    bool found = false;
    try {
      e.getByToken(tok, coll);
      found = true;
    } catch (...) {
      cout << prefix << " Error: no " << name << " rechit data" << endl;
    }
    if (found) {
      for (typename Collection::const_iterator j = coll->begin(); j != coll->end(); ++j) {
        cout << prefix << *j;
        printRecHitAuxInfo(cout, *j, bits_, printPlan1Info);
        cout << endl;
      }
    }
  }
};

HcalRecHitDump::HcalRecHitDump(edm::ParameterSet const& conf)
    : hbhePrefix_(conf.getUntrackedParameter<string>("hbhePrefix", "")),
      hfPrefix_(conf.getUntrackedParameter<string>("hfPrefix", "")),
      hfprePrefix_(conf.getUntrackedParameter<string>("hfprePrefix", "")),
      bits_(conf.getUntrackedParameter<std::vector<int> >("bits")),
      printPlan1Info_(conf.getUntrackedParameter<bool>("printPlan1Info", false)),
      counter_(0) {
  if (!hbhePrefix_.empty())
    tok_hbhe_ = consumes<HBHERecHitCollection>(conf.getParameter<edm::InputTag>("tagHBHE"));
  if (!hfPrefix_.empty())
    tok_hf_ = consumes<HFRecHitCollection>(conf.getParameter<edm::InputTag>("tagHF"));
  if (!hfprePrefix_.empty())
    tok_prehf_ = consumes<HFPreRecHitCollection>(conf.getParameter<edm::InputTag>("tagPreHF"));
}

void HcalRecHitDump::analyze(edm::Event const& e, edm::EventSetup const& c) {
  if (!hbhePrefix_.empty())
    analyzeT<HBHERecHitCollection>(e, tok_hbhe_, "HBHE", hbhePrefix_, printPlan1Info_);
  if (!hfPrefix_.empty())
    analyzeT<HFRecHitCollection>(e, tok_hf_, "HF", hfPrefix_);
  if (!hfprePrefix_.empty())
    analyzeT<HFPreRecHitCollection>(e, tok_prehf_, "PreHF", hfprePrefix_);
  ++counter_;
}

DEFINE_FWK_MODULE(HcalRecHitDump);
