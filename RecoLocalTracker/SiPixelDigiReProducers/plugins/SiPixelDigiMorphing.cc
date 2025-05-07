#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include <boost/algorithm/string.hpp>
#include <bitset>
#include <iterator>

typedef std::pair<uint32_t, uint32_t> range;
typedef std::vector<range> region;

class SiPixelDigiMorphing : public edm::stream::EDProducer<> {
public:
  explicit SiPixelDigiMorphing(const edm::ParameterSet& conf);
  ~SiPixelDigiMorphing() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::Event& e, const edm::EventSetup& c) override;

private:
  edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> tPixelDigi_;
  edm::EDPutTokenT<edm::DetSetVector<PixelDigi>> tPutPixelDigi_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyToken_;

  const std::vector<region> theBarrelRegions_;
  const std::vector<region> theEndcapRegions_;

  const int32_t nrows_;
  const int32_t ncols_;
  const int32_t nrocs_;  // in Phase 1, this is ROCs, but could be any subset of a pixel row
  const int32_t iters_;
  const uint32_t fakeAdc_;

  int32_t ncols_r_;  // number of columns per ROC
  int32_t ksize_;    // kernel size

  std::vector<uint64_t> kernel1_;
  std::vector<uint64_t> kernel2_;
  uint64_t mask_;

  enum MorphOption { kDilate, kErode };

  void morph(uint64_t* const imap, uint64_t* omap, uint64_t* const kernel, MorphOption op) const;
  std::vector<region> parseRegions(const std::vector<std::string>& regionStrings, size_t size);
  bool skipDetId(const TrackerTopology* tTopo,
                 const DetId& detId,
                 const std::vector<region>& theBarrelRegions,
                 const std::vector<region>& theEndcapRegions) const;
};

SiPixelDigiMorphing::SiPixelDigiMorphing(edm::ParameterSet const& conf)
    : tPixelDigi_(consumes(conf.getParameter<edm::InputTag>("src"))),
      tPutPixelDigi_(produces<edm::DetSetVector<PixelDigi>>()),
      trackerTopologyToken_(esConsumes()),
      theBarrelRegions_(parseRegions(conf.getParameter<std::vector<std::string>>("barrelRegions"), 3)),
      theEndcapRegions_(parseRegions(conf.getParameter<std::vector<std::string>>("endcapRegions"), 4)),
      nrows_(conf.getParameter<int32_t>("nrows")),
      ncols_(conf.getParameter<int32_t>("ncols")),
      nrocs_(conf.getParameter<int32_t>("nrocs")),
      iters_(conf.getParameter<int32_t>("iters")),
      fakeAdc_(conf.getParameter<uint32_t>("fakeAdc")) {
  if (ncols_ % nrocs_ == 0) {
    ncols_r_ = ncols_ / nrocs_;
  } else {
    throw cms::Exception("Configuration") << "[SiPixelDigiMorphing]:"
                                          << " number of columns not divisible with"
                                          << " number of ROCs\n";
  }

  if (ncols_r_ + 2 * iters_ <= int(sizeof(uint64_t) * 8)) {
    ksize_ = 2 * iters_ + 1;
  } else {
    throw cms::Exception("Configuration") << "[SiPixelDigiMorphing]:"
                                          << " too many columns per ROC"
                                          << " or too many iterations set\n"
                                          << " Ncol/Nrocs+2*iters should not be"
                                          << " more than " << sizeof(uint64_t) * 8 << "\n";
  }

  std::vector<int32_t> k1(conf.getParameter<std::vector<int32_t>>("kernel1"));
  std::vector<int32_t> k2(conf.getParameter<std::vector<int32_t>>("kernel2"));

  kernel1_.resize(ksize_, 0);
  kernel2_.resize(ksize_, 0);
  mask_ = 0;
  int w = (ncols_r_ + 2 * iters_) / ksize_ + ((ncols_r_ + 2 * iters_) % ksize_ != 0);
  for (int j = 0; j < w; j++) {
    for (int ii = 0; ii < ksize_; ii++) {
      kernel1_[ii] <<= ksize_;
      kernel1_[ii] |= k1[ii];
      kernel2_[ii] <<= ksize_;
      kernel2_[ii] |= k2[ii];
    }
    mask_ <<= ksize_;
    mask_ |= 1;
  }
  mask_ <<= iters_;
}

void SiPixelDigiMorphing::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("src", edm::InputTag("siPixelDigis"));
  // LAYER,LADDER,MODULE (coordinates can also be specified as a range FIRST-LAST where appropriate)
  desc.add<std::vector<std::string>>("barrelRegions", {"1,1-12,1-2", "1,1-12,7-8", "2,1-28,1", "1,1-28,8"});
  // DISK,BLADE,SIDE,PANEL (coordinates can also be specified as a range FIRST-LAST where appropriate)
  desc.add<std::vector<std::string>>("endcapRegions", {});
  desc.add<int32_t>("nrows", 160);
  desc.add<int32_t>("ncols", 416);
  desc.add<int32_t>("nrocs", 8);
  desc.add<int32_t>("iters", 1);
  desc.add<std::vector<int32_t>>("kernel1", {7, 7, 7});
  desc.add<std::vector<int32_t>>("kernel2", {2, 7, 2});
  desc.add<uint32_t>("fakeAdc", 100);

  descriptions.addWithDefaultLabel(desc);
}

void SiPixelDigiMorphing::produce(edm::Event& e, const edm::EventSetup& es) {
  auto const& inputDigi = e.get(tPixelDigi_);

  auto outputDigis = std::make_unique<edm::DetSetVector<PixelDigi>>();

  // use the TrackerTopology for the layer/disk number, etc.
  const TrackerTopology* tTopo = &es.getData(trackerTopologyToken_);

  const int rocSize = nrows_ + 2 * iters_;
  const int arrSize = nrocs_ * rocSize;

  uint64_t imap[arrSize];
  uint64_t map1[arrSize];
  uint64_t map2[arrSize];

  for (auto const& ds : inputDigi) {
    auto rawId = ds.detId();

    const DetId detId(rawId);

    // skip DetIds for which digi morphing has not been requested
    if (skipDetId(tTopo, detId, theBarrelRegions_, theEndcapRegions_)) {
      outputDigis->insert(ds);
      continue;
    }

    edm::DetSet<PixelDigi>* detDigis = nullptr;
    detDigis = &(outputDigis->find_or_insert(rawId));

    memset(imap, 0, arrSize * sizeof(uint64_t));
    for (auto const& di : ds) {
      int r = int(di.column()) / ncols_r_;
      int c = int(di.column()) % ncols_r_;
      imap[r * rocSize + di.row() + iters_] |= uint64_t(1) << (c + iters_);
      if (r > 0 && c < iters_) {
        imap[(r - 1) * rocSize + di.row() + iters_] |= uint64_t(1) << (c + ncols_r_ + iters_);
      } else if (++r < nrocs_ && c >= ncols_r_ - iters_) {
        imap[r * rocSize + di.row() + iters_] |= uint64_t(1) << (c - ncols_r_ + iters_);
      }
      (*detDigis).data.emplace_back(di.row(), di.column(), di.adc(), 0);
    }

    std::memcpy(map1, imap, arrSize * sizeof(uint64_t));
    memset(map2, 0, arrSize * sizeof(uint64_t));

    morph(map1, map2, kernel1_.data(), kDilate);
    morph(map2, map1, kernel2_.data(), kErode);

    uint64_t* i = imap + iters_;
    uint64_t* o = map1 + iters_;
    for (int roc = 0; roc < nrocs_; roc++, i += 2 * iters_, o += 2 * iters_) {
      for (int row = 0; row < nrows_; row++, i++, o++) {
        if (*o == 0)
          continue;
        *o >>= iters_;
        *i >>= iters_;
        for (int col = 0; col < ncols_r_; col++, (*i) >>= 1, (*o) >>= 1) {
          if (*o == 0)
            break;
          if (((*i) & uint64_t(1)) == 1)
            continue;
          if (((*o) & uint64_t(1)) == 1) {
            (*detDigis).data.emplace_back(row, roc * ncols_r_ + col, fakeAdc_, 1);
          }
        }
      }
    }
  }
  e.put(tPutPixelDigi_, std::move(outputDigis));
}

void SiPixelDigiMorphing::morph(uint64_t* const imap, uint64_t* omap, uint64_t* const kernel, MorphOption op) const {
  uint64_t* i[ksize_];          // i(nput)
  uint64_t* o = omap + iters_;  // o(output)
  unsigned char valid = 0;
  unsigned char const validMask = (1 << ksize_) - 1;
  uint64_t m[ksize_];  // m(ask)

  for (int ii = 0; ii < ksize_; ii++) {
    i[ii] = imap + ii;
    valid = (valid << 1) | (*i[ii] != 0);
    m[ii] = mask_ << ii;
  }

  for (int roc = 0; roc < nrocs_; roc++, o += 2 * iters_) {
    for (int row = 0; row < nrows_; row++, o++) {
      if ((valid & validMask) != 0) {
        for (int jj = 0; jj < ksize_; jj++) {
          for (int ii = 0; ii < ksize_; ii++) {
            uint64_t v = (*i[ii]) & (kernel[ii] << jj);  // v(ector)
            if (op == kErode)
              v ^= (kernel[ii] << jj);
            uint64_t vv = v;  // vv(vector bit - shifted and contracted)
            for (int b = 1; b < ksize_; b++)
              vv |= (v >> b);
            *o |= ((vv << iters_) & m[jj]);
          }
          if (op == kErode)
            *o ^= m[jj];
        }
      }
      for (int ii = 0; ii < ksize_; ii++)
        i[ii]++;
      valid = (valid << 1) | (*i[ksize_ - 1] != 0);
    }
    for (int ii = 0; ii < ksize_; ii++) {
      i[ii] += 2 * iters_;
      valid = (valid << 1) | (*i[ii] != 0);
    }
  }
}

std::vector<region> SiPixelDigiMorphing::parseRegions(const std::vector<std::string>& regionStrings, size_t size) {
  std::vector<region> regions;

  for (auto const& str : regionStrings) {
    region reg;

    std::vector<std::string> ranges;
    boost::split(ranges, str, boost::is_any_of(","));

    if (ranges.size() != size) {
      throw cms::Exception("Configuration") << "[SiPixelDigiMorphing]:"
                                            << " invalid number of coordinates provided in " << str << " (" << size
                                            << " expected, " << ranges.size() << " provided)\n";
    }

    for (auto const& r : ranges) {
      std::vector<std::string> limits;
      boost::split(limits, r, boost::is_any_of("-"));

      try {
        // if range specified
        if (limits.size() > 1) {
          reg.push_back(std::make_pair(std::stoi(limits.at(0)), std::stoi(limits.at(1))));
          // otherwise store single value as a range
        } else {
          reg.push_back(std::make_pair(std::stoi(limits.at(0)), std::stoi(limits.at(0))));
        }
      } catch (...) {
        throw cms::Exception("Configuration") << "[SiPixelDigiMorphing]:"
                                              << " invalid coordinate value provided in " << str << "\n";
      }
    }
    regions.push_back(reg);
  }

  return regions;
}

// apply regional digi morphing logic
bool SiPixelDigiMorphing::skipDetId(const TrackerTopology* tTopo,
                                    const DetId& detId,
                                    const std::vector<region>& theBarrelRegions,
                                    const std::vector<region>& theEndcapRegions) const {
  // barrel
  if (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)) {
    // no barrel region specified
    if (theBarrelRegions.empty()) {
      return true;
    } else {
      uint32_t layer = tTopo->pxbLayer(detId.rawId());
      uint32_t ladder = tTopo->pxbLadder(detId.rawId());
      uint32_t module = tTopo->pxbModule(detId.rawId());

      bool inRegion = false;

      for (auto const& reg : theBarrelRegions) {
        if ((layer >= reg.at(0).first && layer <= reg.at(0).second) &&
            (ladder >= reg.at(1).first && ladder <= reg.at(1).second) &&
            (module >= reg.at(2).first && module <= reg.at(2).second)) {
          inRegion = true;
          break;
        }
      }

      return !inRegion;
    }
    // endcap
  } else {
    // no endcap region specified
    if (theEndcapRegions.empty()) {
      return true;
    } else {
      uint32_t disk = tTopo->pxfDisk(detId.rawId());
      uint32_t blade = tTopo->pxfBlade(detId.rawId());
      uint32_t side = tTopo->pxfSide(detId.rawId());
      uint32_t panel = tTopo->pxfPanel(detId.rawId());

      bool inRegion = false;

      for (auto const& reg : theEndcapRegions) {
        if ((disk >= reg.at(0).first && disk <= reg.at(0).second) &&
            (blade >= reg.at(1).first && blade <= reg.at(1).second) &&
            (side >= reg.at(2).first && side <= reg.at(2).second) &&
            (panel >= reg.at(3).first && panel <= reg.at(3).second)) {
          inRegion = true;
          break;
        }
      }

      return !inRegion;
    }
  }
}

DEFINE_FWK_MODULE(SiPixelDigiMorphing);
