// system includes
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// user includes
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/**
    @file EventFilter/SiStripRawToDigi/test/plugins/SiStripTrivialDigiSource.h
    @class SiStripTrivialDigiSource

    @brief Creates a DetSetVector of SiStripDigis created using random
    number generators and attaches the collection to the Event. Allows
    to test the final DigiToRaw and RawToDigi converters.  
*/
class SiStripTrivialDigiSource : public edm::global::EDProducer<> {
public:
  SiStripTrivialDigiSource(const edm::ParameterSet&);
  ~SiStripTrivialDigiSource();

  virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::ESGetToken<SiStripFedCabling, SiStripFedCablingRcd> esTokenCabling_;
  const float meanOcc_;
  const float rmsOcc_;
  const int ped_;
  const bool raw_;
  const bool useFedKey_;
};

// -----------------------------------------------------------------------------
//
SiStripTrivialDigiSource::SiStripTrivialDigiSource(const edm::ParameterSet& pset)
    : esTokenCabling_(esConsumes()),
      meanOcc_(pset.getUntrackedParameter<double>("MeanOccupancy", 1.)),
      rmsOcc_(pset.getUntrackedParameter<double>("RmsOccupancy", 0.1)),
      ped_(pset.getUntrackedParameter<int>("PedestalLevel", 100)),
      raw_(pset.getUntrackedParameter<bool>("FedRawDataMode", false)),
      useFedKey_(pset.getUntrackedParameter<bool>("UseFedKey", false)) {
  LogTrace("TrivialDigiSource") << "[SiStripTrivialDigiSource::" << __func__ << "]"
                                << " Constructing object...";
  produces<edm::DetSetVector<SiStripDigi>>();
}

// -----------------------------------------------------------------------------
//
SiStripTrivialDigiSource::~SiStripTrivialDigiSource() {
  LogTrace("TrivialDigiSource") << "[SiStripTrivialDigiSource::" << __func__ << "]"
                                << " Destructing object...";
}

// -----------------------------------------------------------------------------
//
void SiStripTrivialDigiSource::produce(edm::StreamID, edm::Event& event, const edm::EventSetup& setup) const {
  LogTrace("TrivialDigiSource") << "[SiStripRawToDigiModule::" << __func__ << "]"
                                << " Analyzing run/event " << event.id().run() << "/" << event.id().event();

  // Retrieve cabling
  edm::ESHandle<SiStripFedCabling> cabling = setup.getHandle(esTokenCabling_);

  // Temp container
  typedef std::vector<edm::DetSet<SiStripDigi>> digi_work_vector;
  digi_work_vector zero_suppr_vector;

  // Some init
  uint32_t nchans = 0;
  uint32_t ndigis = 0;

  // Retrieve fed ids
  auto fed_ids = cabling->fedIds();

  // Iterate through fed ids and channels
  for (auto ifed = fed_ids.begin(); ifed != fed_ids.end(); ifed++) {
    auto conns = cabling->fedConnections(*ifed);
    for (auto iconn = conns.begin(); iconn != conns.end(); iconn++) {
      // Build FED key
      uint32_t fed_key = ((iconn->fedId() & sistrip::invalid_) << 16) | (iconn->fedCh() & sistrip::invalid_);

      // Determine key (FED key or DetId) to index DSV
      uint32_t key = useFedKey_ ? fed_key : iconn->detId();

      // Determine APV pair number
      uint16_t ipair = useFedKey_ ? 0 : iconn->apvPairNumber();

      // Check key is non-zero and valid
      if (!key || (key == sistrip::invalid32_)) {
        continue;
      }

      // Random number of digis
      double tmp = 0.;
      float rdm = 2.56 * CLHEP::RandGauss::shoot(meanOcc_, rmsOcc_);
      bool extra = (CLHEP::RandFlat::shoot() > modf(rdm, &tmp));
      uint16_t ndigi = static_cast<uint16_t>(rdm) + static_cast<uint16_t>(extra);

      // Create DetSet
      if (zero_suppr_vector.empty()) {
        zero_suppr_vector.reserve(40000);
      }
      zero_suppr_vector.push_back(edm::DetSet<SiStripDigi>(key));
      edm::DetSet<SiStripDigi>& zs = zero_suppr_vector.back();
      zs.data.reserve(768);

      // Remember strips used
      std::vector<uint16_t> used_strips;
      used_strips.reserve(ndigi);

      // Create digis
      uint16_t idigi = 0;
      while (idigi < ndigi) {
        // Random values
        uint16_t str = static_cast<uint16_t>(256. * CLHEP::RandFlat::shoot());
        uint16_t adc = static_cast<uint16_t>(256. * CLHEP::RandFlat::shoot());

        // Generate and check strip number
        uint16_t nstrips = iconn->nDetStrips();
        uint16_t strip = str + 256 * ipair;
        if (strip >= nstrips) {
          continue;
        }

        // Create digi object
        std::vector<uint16_t>::iterator iter = find(used_strips.begin(), used_strips.end(), strip);
        if (iter == used_strips.end() && adc) {  // require non-zero adc!
          uint16_t level = raw_ ? ped_ + adc : adc;
          zs.data.push_back(SiStripDigi(strip, level));
          used_strips.push_back(strip);
          ndigis++;
          idigi++;
        }
      }

      // Populate DetSet with remaining "raw" digis
      if (raw_) {
        for (uint16_t istr = 256 * ipair; istr < 256 * (ipair + 1); ++istr) {
          if (find(used_strips.begin(), used_strips.end(), istr) == used_strips.end()) {
            zs.data.push_back(SiStripDigi(istr, ped_));
          }
        }
      }
    }
  }

  // Create final DetSetVector container
  auto collection = std::make_unique<edm::DetSetVector<SiStripDigi>>();

  // Populate final DetSetVector container with ZS data
  if (!zero_suppr_vector.empty()) {
    std::sort(zero_suppr_vector.begin(), zero_suppr_vector.end());
    std::vector<edm::DetSet<SiStripDigi>> sorted_and_merged;
    sorted_and_merged.reserve(zero_suppr_vector.size());

    edm::det_id_type old_id = 0;
    std::vector<edm::DetSet<SiStripDigi>>::iterator ii = zero_suppr_vector.begin();
    std::vector<edm::DetSet<SiStripDigi>>::iterator jj = zero_suppr_vector.end();
    for (; ii != jj; ++ii) {
      if (old_id == ii->detId()) {
        sorted_and_merged.back().data.insert(sorted_and_merged.back().end(), ii->begin(), ii->end());
      } else {
        sorted_and_merged.push_back(*ii);
        old_id = ii->detId();
      }
    }

    std::vector<edm::DetSet<SiStripDigi>>::iterator iii = sorted_and_merged.begin();
    std::vector<edm::DetSet<SiStripDigi>>::iterator jjj = sorted_and_merged.end();
    for (; iii != jjj; ++iii) {
      std::sort(iii->begin(), iii->end());
    }

    edm::DetSetVector<SiStripDigi> zero_suppr_dsv(sorted_and_merged, true);
    collection->swap(zero_suppr_dsv);
  }

  // Put collection in event
  event.put(std::move(collection));

  // Some debug
  if (edm::isDebugEnabled() && nchans) {
    std::stringstream ss;
    ss << "[SiStripTrivialDigiSource::" << __func__ << "]"
       << " Generated " << ndigis << " digis for " << nchans << " channels with a mean occupancy of " << std::dec
       << std::setprecision(2) << (1. / 2.56) * (float)ndigis / (float)nchans << " %";
    LogTrace("TrivialDigiSource") << ss.str();
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripTrivialDigiSource);
