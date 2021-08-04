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

#include <bitset>

class SiPixelDigiMorphing : public edm::stream::EDProducer<> {
public:
  explicit SiPixelDigiMorphing(const edm::ParameterSet& conf);
  ~SiPixelDigiMorphing() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::Event& e, const edm::EventSetup& c) override;

private:
  edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> tPixelDigi_;
  edm::EDPutTokenT<edm::DetSetVector<PixelDigi>> tPutPixelDigi_;

  int32_t nrows_;
  int32_t ncols_;
  int32_t nrocs_;  // in Phase 1, this is ROCs, but could be any subset of a pixel row
  int32_t ncols_r_;

  int32_t iters_;
  int32_t ksize_;

  std::vector<uint64_t> kernel1_;
  std::vector<uint64_t> kernel2_;
  uint64_t mask_;

  enum MorphOption { kDilate, kErode };

  uint32_t fakeAdc_;

  void morph(uint64_t* imap, uint64_t* omap, uint64_t* kernel, MorphOption op);
};

SiPixelDigiMorphing::SiPixelDigiMorphing(edm::ParameterSet const& conf)
    : tPutPixelDigi_(produces<edm::DetSetVector<PixelDigi>>()),
      nrows_(conf.getParameter<int32_t>("nrows")),
      ncols_(conf.getParameter<int32_t>("ncols")),
      nrocs_(conf.getParameter<int32_t>("nrocs")),
      iters_(conf.getParameter<int32_t>("iters")),
      fakeAdc_(conf.getParameter<uint32_t>("fakeAdc")) {
  tPixelDigi_ = consumes<edm::DetSetVector<PixelDigi>>(conf.getParameter<edm::InputTag>("src"));

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
                                          << " more than " << sizeof(uint64_t) << "\n";
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

SiPixelDigiMorphing::~SiPixelDigiMorphing() = default;

void SiPixelDigiMorphing::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("src", edm::InputTag("siPixelDigis"));
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
  edm::Handle<edm::DetSetVector<PixelDigi>> inputDigi;
  e.getByToken(tPixelDigi_, inputDigi);

  auto outputDigis = std::make_unique<edm::DetSetVector<PixelDigi>>();

  uint64_t imap[nrocs_ * (nrows_ + 2 * iters_)];
  uint64_t map1[nrocs_ * (nrows_ + 2 * iters_)];
  uint64_t map2[nrocs_ * (nrows_ + 2 * iters_)];

  edm::DetSetVector<PixelDigi>::const_iterator DSViter = (*inputDigi).begin();
  for (; DSViter != (*inputDigi).end(); DSViter++) {
    auto rawId = DSViter->detId();
    edm::DetSet<PixelDigi>* detDigis = nullptr;
    detDigis = &(outputDigis->find_or_insert(rawId));

    edm::DetSet<PixelDigi>::const_iterator di;
    memset(imap, 0, nrocs_ * (nrows_ + 2 * iters_) * sizeof(uint64_t));
    for (di = DSViter->data.begin(); di != DSViter->data.end(); di++) {
      int r = int(di->column()) / ncols_r_;
      int c = int(di->column()) % ncols_r_;
      imap[r * (nrows_ + 2 * iters_) + di->row() + iters_] |= uint64_t(1) << (c + iters_);
      if (r > 0 && c < iters_) {
        imap[(r - 1) * (nrows_ + 2 * iters_) + di->row() + iters_] |= uint64_t(1) << (c + ncols_r_ + iters_);
      } else if (++r < nrocs_ && c >= ncols_r_ - iters_) {
        imap[r * (nrows_ + 2 * iters_) + di->row() + iters_] |= uint64_t(1) << (c - ncols_r_ + iters_);
      }
      (*detDigis).data.emplace_back(di->row(), di->column(), di->adc(), 0);
    }

    std::memcpy(map1, imap, nrocs_ * (nrows_ + 2 * iters_) * sizeof(uint64_t));
    memset(map2, 0, nrocs_ * (nrows_ + 2 * iters_) * sizeof(uint64_t));

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

void SiPixelDigiMorphing::morph(uint64_t* imap, uint64_t* omap, uint64_t* kernel, MorphOption op) {
  uint64_t* i[ksize_];
  uint64_t* o = omap + iters_;
  unsigned char valid = 0;
  unsigned char validMask = (1 << ksize_) - 1;
  uint64_t m[ksize_];

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
            uint64_t v = (*i[ii]) & (kernel[ii] << jj);
            if (op == kErode)
              v ^= (kernel[ii] << jj);
            uint64_t vv = v;
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

DEFINE_FWK_MODULE(SiPixelDigiMorphing);
