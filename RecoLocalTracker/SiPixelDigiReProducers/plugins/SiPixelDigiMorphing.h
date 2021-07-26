#ifndef RecoLocalTracker_SiPixelDigiMorphing_SiPixelDigiMorphing_h
#define RecoLocalTracker_SiPixelDigiMorphing_SiPixelDigiMorphing_h

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

class SiPixelDigiMorphing : public edm::stream::EDProducer<> {
public:
  explicit SiPixelDigiMorphing(const edm::ParameterSet& conf);
  ~SiPixelDigiMorphing() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::Event& e, const edm::EventSetup& c) override;

private:
  edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> tPixelDigi;
  edm::EDPutTokenT<edm::DetSetVector<PixelDigi>> tPutPixelDigi;

  int32_t nrows_;
  int32_t ncols_;
  int32_t nrocs_;  // in Phase 1, this is ROCs, but could be any subset of a pixel row
  int32_t ncols_r_;

  int32_t iters_;
  int32_t ksize_;

  std::vector<uint64_t> kernel1_;
  std::vector<uint64_t> kernel2_;
  uint64_t mask_;

  enum MorphOption { Dilate, Erode };

  uint32_t fakeAdc_;

  void morph(uint64_t* imap, uint64_t* omap, uint64_t* kernel, MorphOption op);
};

#endif
