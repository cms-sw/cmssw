// system includes
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

// user includes
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripEventSummary.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/**
   @class SiStripDigiAnalyzer
   @brief Simple class that analyzes Digis produced by RawToDigi unpacker
*/

class SiStripTrivialDigiAnalysis {
public:
  /** Default constructor. */
  SiStripTrivialDigiAnalysis()
      : events_(0), feds_(0), channels_(0), strips_(0), digis_(0), size_(1024), pos_(size_ + 1, 0), adc_(size_ + 1, 0) {}

  /** Pipes collected statistics to stringstream. */
  void print(std::stringstream&);

  // setters
  inline void pos(const uint16_t& pos);
  inline void adc(const uint16_t& adc);

  // getters
  inline const std::vector<uint16_t>& pos();
  inline const std::vector<uint16_t>& adc();

public:
  uint32_t events_;
  uint32_t feds_;
  uint32_t channels_;
  uint32_t strips_;
  uint32_t digis_;

  const uint16_t size_;

private:
  std::vector<uint16_t> pos_;
  std::vector<uint16_t> adc_;
};

class SiStripDigiAnalyzer : public edm::one::EDAnalyzer<> {
public:
  SiStripDigiAnalyzer(const edm::ParameterSet&);
  ~SiStripDigiAnalyzer();

  void beginJob();
  void analyze(const edm::Event&, const edm::EventSetup&);
  void endJob();

private:
  const edm::ESGetToken<SiStripFedCabling, SiStripFedCablingRcd> esTokenCabling_;
  std::string inputModuleLabel_;

  SiStripTrivialDigiAnalysis anal_;
  SiStripTrivialDigiAnalysis vr_p;
  SiStripTrivialDigiAnalysis pr_p;
  SiStripTrivialDigiAnalysis sm_p;
  SiStripTrivialDigiAnalysis zs_p;
  SiStripTrivialDigiAnalysis vr_r;
  SiStripTrivialDigiAnalysis pr_r;
  SiStripTrivialDigiAnalysis sm_r;
  SiStripTrivialDigiAnalysis zs_r;
};

void SiStripTrivialDigiAnalysis::pos(const uint16_t& pos) {
  if (pos < size_) {
    pos_[pos]++;
  } else {
    pos_[size_]++;
  }
}

void SiStripTrivialDigiAnalysis::adc(const uint16_t& adc) {
  if (adc < size_) {
    adc_[adc]++;
  } else {
    adc_[size_]++;
  }
}

const std::vector<uint16_t>& SiStripTrivialDigiAnalysis::pos() { return pos_; }
const std::vector<uint16_t>& SiStripTrivialDigiAnalysis::adc() { return adc_; }

using namespace std;

void SiStripTrivialDigiAnalysis::print(stringstream& ss) {
  ss << "  [SiStripTrivialDigiAnalysis::print]"
     << " events: " << events_ << " feds: " << feds_ << " channels: " << channels_ << " strips: " << strips_
     << " digis: " << digis_;
  // Signal distribution (strip position vs frequency)
  ss << "\n  strip: ";
  for (uint16_t ii = 0; ii < size_; ii += (size_ / 16)) {
    ss << setw(4) << ii << " ";
  }
  ss << "ovrflw";
  ss << "\n  freq : ";
  for (uint16_t ii = 0; ii < size_; ii += (size_ / 16)) {
    ss << setw(4) << pos_[ii] << " ";
  }
  ss << "  " << setw(4) << pos_.back();
  // Signal landau (ADC counts vs frequency)
  ss << "\n  adc  : ";
  for (uint16_t ii = 0; ii < size_; ii += (size_ / 16)) {
    ss << setw(4) << ii << " ";
  }
  ss << "ovrflw";
  ss << "\n  freq : ";
  for (uint16_t ii = 0; ii < size_; ii += (size_ / 16)) {
    ss << setw(4) << adc_[ii] << " ";
  }
  ss << "  " << setw(4) << adc_.back();
  // Misc
  uint16_t cntr = 0;
  uint32_t tmp = 0;
  ss << "\n  adc/freq: ";
  for (uint16_t ii = 0; ii < size_; ii++) {
    if (adc_[ii]) {
      if (cntr < 8) {
        ss << ii << "/" << adc_[ii] << ", ";
        cntr++;
      }
      tmp += adc_[ii];
    }
  }
  ss << "ovrflw: " << adc_.back();
  ss << ", total: " << tmp;
}

// -----------------------------------------------------------------------------
//
SiStripDigiAnalyzer::SiStripDigiAnalyzer(const edm::ParameterSet& pset)
    : esTokenCabling_(esConsumes()), inputModuleLabel_(pset.getParameter<string>("InputModuleLabel")) {
  consumes<edm::DetSetVector<SiStripRawDigi> >(edm::InputTag(inputModuleLabel_, "VirginRaw"));
  consumes<edm::DetSetVector<SiStripRawDigi> >(edm::InputTag(inputModuleLabel_, "ProcessedRaw"));
  consumes<edm::DetSetVector<SiStripRawDigi> >(edm::InputTag(inputModuleLabel_, "ScopeMode"));
  consumes<edm::DetSetVector<SiStripDigi> >(edm::InputTag(inputModuleLabel_, "ZeroSuppressed"));
  consumes<SiStripEventSummary>(inputModuleLabel_);
  LogDebug("SiStripDigiAnalyzer") << "[SiStripDigiAnalyzer::SiStripDigiAnalyzer]"
                                  << " Constructing object...";
}

// -----------------------------------------------------------------------------
//
SiStripDigiAnalyzer::~SiStripDigiAnalyzer() {
  LogDebug("SiStripDigiAnalyzer") << "[SiStripDigiAnalyzer::~SiStripDigiAnalyzer]"
                                  << " Destructing object...";
}

// -----------------------------------------------------------------------------
//
void SiStripDigiAnalyzer::beginJob() { LogDebug("SiStripDigiAnalyzer") << "[SiStripDigiAnalyzer::beginJob]"; }

// -----------------------------------------------------------------------------
//
void SiStripDigiAnalyzer::endJob() {
  stringstream ss;
  ss << "PSEUDO DIGI ANALYSIS:"
     << "\n";
  anal_.print(ss);
  ss << "\n";
  ss << "REAL DIGI (VIRGIN RAW) ANALYSIS:"
     << "\n";
  vr_r.print(ss);
  ss << "\n";
  ss << "REAL DIGI (PROCESSED RAW) ANALYSIS:"
     << "\n";
  pr_r.print(ss);
  ss << "\n";
  ss << "REAL DIGI (SCOPE MODE) ANALYSIS:"
     << "\n";
  sm_r.print(ss);
  ss << "\n";
  ss << "REAL DIGI (ZERO SUPPR) ANALYSIS:"
     << "\n";
  zs_r.print(ss);
  ss << "\n";
  LogDebug("SiStripDigiAnalyzer") << ss.str();
}

// -----------------------------------------------------------------------------
//
void SiStripDigiAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  LogDebug("SiStripDigiAnalyzer") << "[" << __PRETTY_FUNCTION__ << "]"
                                  << " Analyzing run " << event.id().run() << " and event " << event.id().event();

  // Retrieve FED (reatout) and FEC (control) cabling
  edm::ESHandle<SiStripFedCabling> fed_cabling = setup.getHandle(esTokenCabling_);

  // Retrieve "real" digis
  edm::Handle<edm::DetSetVector<SiStripRawDigi> > vr;
  edm::Handle<edm::DetSetVector<SiStripRawDigi> > pr;
  edm::Handle<edm::DetSetVector<SiStripRawDigi> > sm;
  edm::Handle<edm::DetSetVector<SiStripDigi> > zs;
  event.getByLabel(inputModuleLabel_, "VirginRaw", vr);
  event.getByLabel(inputModuleLabel_, "ProcessedRaw", pr);
  event.getByLabel(inputModuleLabel_, "ScopeMode", sm);
  event.getByLabel(inputModuleLabel_, "ZeroSuppressed", zs);

  // Retrieve SiStripEventSummary
  edm::Handle<SiStripEventSummary> summary;
  event.getByLabel(inputModuleLabel_, summary);

  // Analyse digis
  anal_.events_++;
  vr_r.events_++;
  pr_r.events_++;
  sm_r.events_++;
  zs_r.events_++;
  auto ifed = fed_cabling->fedIds().begin();
  for (; ifed != fed_cabling->fedIds().end(); ++ifed) {
    anal_.feds_++;
    vr_r.feds_++;
    pr_r.feds_++;
    sm_r.feds_++;
    zs_r.feds_++;
    for (uint16_t ichan = 0; ichan < sistrip::FEDCH_PER_FED; ichan++) {
      anal_.channels_++;
      vr_r.channels_++;
      pr_r.channels_++;
      sm_r.channels_++;
      zs_r.channels_++;

      // Analyse digis

      uint32_t key = SiStripFedKey(*ifed, SiStripFedKey::feUnit(ichan), SiStripFedKey::feChan(ichan)).key();

      vector<edm::DetSet<SiStripRawDigi> >::const_iterator raw;
      vector<edm::DetSet<SiStripDigi> >::const_iterator digis;

      // virgin raw
      raw = vr->find(key);
      if (raw != vr->end()) {
        for (uint16_t istrip = 0; istrip < raw->size(); istrip++) {
          if (raw->data[istrip].adc()) {
            vr_r.strips_++;
            vr_r.pos(istrip);
            vr_r.adc(raw->data[istrip].adc());
          }
        }
      }

      // processed raw
      raw = pr->find(key);
      if (raw != pr->end()) {
        for (uint16_t istrip = 0; istrip < raw->size(); istrip++) {
          if (raw->data[istrip].adc()) {
            pr_r.strips_++;
            pr_r.pos(istrip);
            pr_r.adc(raw->data[istrip].adc());
          }
        }
      }

      // scope mode
      raw = sm->find(key);
      if (raw != sm->end()) {
        for (uint16_t istrip = 0; istrip < raw->size(); istrip++) {
          if (raw->data[istrip].adc()) {
            sm_r.strips_++;
            sm_r.pos(istrip);
            sm_r.adc(raw->data[istrip].adc());
          }
        }
      }

      // scope mode
      digis = zs->find(key);
      if (digis != zs->end()) {
        for (uint16_t iadc = 0; iadc < digis->size(); iadc++) {
          if (digis->data[iadc].adc()) {
            zs_r.strips_++;
            zs_r.pos(digis->data[iadc].strip());
            zs_r.adc(digis->data[iadc].adc());
          }
        }
      }
    }  // channel loop
  }    // fed loop
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripDigiAnalyzer);
