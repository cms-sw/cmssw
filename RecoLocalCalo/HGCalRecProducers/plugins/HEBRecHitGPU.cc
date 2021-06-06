#include <iostream>
#include <string>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/ContextState.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/MessageLogger.h"

#include "RecoLocalCalo/HGCalRecProducers/plugins/KernelManagerHGCalRecHit.h"
#include "CUDADataFormats/HGCal/interface/HGCRecHitGPUProduct.h"
#include "CUDADataFormats/HGCal/interface/HGCUncalibRecHitDevice.h"
#include "CUDADataFormats/HGCal/interface/HGCUncalibRecHitHost.h"

class HEBRecHitGPU : public edm::stream::EDProducer<> {
public:
  explicit HEBRecHitGPU(const edm::ParameterSet &ps);
  ~HEBRecHitGPU() override;
  void beginRun(edm::Run const &, edm::EventSetup const &) override;

  void produce(edm::Event &, const edm::EventSetup &) override;

private:
  edm::EDGetTokenT<HGChebUncalibratedRecHitCollection> uncalibRecHitCPUToken_;
  edm::EDPutTokenT<cms::cuda::Product<HGCRecHitGPUProduct>> recHitGPUToken_;

  std::unique_ptr<HGChebRecHitCollection> rechits_;

  //constants
  HGChebUncalibRecHitConstantData cdata_;
  HGCConstantVectorData vdata_;

  //memory
  std::string assert_error_message_(std::string, const size_t &);
  void assert_sizes_constants_(const HGCConstantVectorData &);

  //conditions (geometry, topology, ...)
  std::unique_ptr<hgcal::RecHitTools> tools_;

  //data processing
  void convert_collection_data_to_soa_(const uint32_t &, const HGChebUncalibratedRecHitCollection &);
  void convert_constant_data_(KernelConstantData<HGChebUncalibRecHitConstantData> *);

  HGCRecHitGPUProduct prod_;
  HGCUncalibRecHitDevice d_uncalib_;
  HGCUncalibRecHitHost<HGChebUncalibratedRecHitCollection> h_uncalib_;

  KernelConstantData<HGChebUncalibRecHitConstantData> *kcdata_;
};

HEBRecHitGPU::HEBRecHitGPU(const edm::ParameterSet &ps)
    : uncalibRecHitCPUToken_{consumes<HGCUncalibratedRecHitCollection>(
          ps.getParameter<edm::InputTag>("HGCHEBUncalibRecHitsTok"))},
      recHitGPUToken_{produces<cms::cuda::Product<HGCRecHitGPUProduct>>()} {
  cdata_.keV2DIGI_ = ps.getParameter<double>("HGCHEB_keV2DIGI");
  cdata_.noise_MIP_ = ps.getParameter<edm::ParameterSet>("HGCHEB_noise_MIP").getParameter<double>("noise_MIP");
  vdata_.weights_ = ps.getParameter<std::vector<double>>("weights");
  cdata_.uncalib2GeV_ = 1e-6 / cdata_.keV2DIGI_;
  cdata_.layerOffset_ = 28;
  assert_sizes_constants_(vdata_);

  kcdata_ = new KernelConstantData<HGChebUncalibRecHitConstantData>(cdata_, vdata_);
  convert_constant_data_(kcdata_);

  tools_ = std::make_unique<hgcal::RecHitTools>();
}

HEBRecHitGPU::~HEBRecHitGPU() { delete kcdata_; }

std::string HEBRecHitGPU::assert_error_message_(std::string var, const size_t &s) {
  std::string str1 = "The '";
  std::string str2 = "' array must be of size ";
  std::string str3 = " to hold the configuration data.";
  return str1 + var + str2 + std::to_string(s) + str3;
}

void HEBRecHitGPU::assert_sizes_constants_(const HGCConstantVectorData &vd) {
  if (vdata_.weights_.size() != HGChebUncalibRecHitConstantData::heb_weights)
    edm::LogError("WrongSize") << this->assert_error_message_("weights", vdata_.fCPerMIP_.size());
}

void HEBRecHitGPU::beginRun(edm::Run const &, edm::EventSetup const &setup) {}

void HEBRecHitGPU::produce(edm::Event &event, const edm::EventSetup &setup) {
  cms::cuda::ScopedContextProduce ctx{event.streamID()};

  const auto &hits = event.get(uncalibRecHitCPUToken_);
  unsigned int nhits(hits.size());
  rechits_ = std::make_unique<HGCRecHitCollection>();

  if (nhits == 0)
    edm::LogError("HEBRecHitGPU") << "WARNING: no input hits!";

  prod_ = HGCRecHitGPUProduct(nhits, ctx.stream());
  d_uncalib_ = HGCUncalibRecHitDevice(nhits, ctx.stream());
  h_uncalib_ = HGCUncalibRecHitHost<HGChebUncalibratedRecHitCollection>(nhits, hits, ctx.stream());

  KernelManagerHGCalRecHit km(h_uncalib_.get(), d_uncalib_.get(), prod_.get());
  km.run_kernels(kcdata_, ctx.stream());

  ctx.emplace(event, recHitGPUToken_, std::move(prod_));
}

void HEBRecHitGPU::convert_constant_data_(KernelConstantData<HGChebUncalibRecHitConstantData> *kcdata) {
  for (size_t i = 0; i < kcdata->vdata_.weights_.size(); ++i)
    kcdata->data_.weights_[i] = kcdata->vdata_.weights_[i];
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HEBRecHitGPU);
