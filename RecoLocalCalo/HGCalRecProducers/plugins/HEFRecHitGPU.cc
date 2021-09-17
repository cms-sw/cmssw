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

class HEFRecHitGPU : public edm::stream::EDProducer<> {
public:
  explicit HEFRecHitGPU(const edm::ParameterSet &ps);
  ~HEFRecHitGPU() override;
  void beginRun(edm::Run const &, edm::EventSetup const &) override;

  void produce(edm::Event &, const edm::EventSetup &) override;

private:
  edm::EDGetTokenT<HGChefUncalibratedRecHitCollection> uncalibRecHitCPUToken_;
  edm::EDPutTokenT<cms::cuda::Product<HGCRecHitGPUProduct>> recHitGPUToken_;

  std::unique_ptr<HGChefRecHitCollection> rechits_;

  //constants
  HGChefUncalibRecHitConstantData cdata_;
  HGCConstantVectorData vdata_;

  //memory
  std::string assert_error_message_(std::string, const size_t &, const size_t &);
  void assert_sizes_constants_(const HGCConstantVectorData &);

  //conditions (geometry, topology, ...)
  std::unique_ptr<hgcal::RecHitTools> tools_;

  //data processing
  void convert_collection_data_to_soa_(const uint32_t &, const HGChefUncalibratedRecHitCollection &);
  void convert_constant_data_(KernelConstantData<HGChefUncalibRecHitConstantData> *);

  HGCRecHitGPUProduct prod_;
  HGCUncalibRecHitDevice d_uncalib_;
  HGCUncalibRecHitHost<HGChefUncalibratedRecHitCollection> h_uncalib_;

  KernelConstantData<HGChefUncalibRecHitConstantData> *kcdata_;
};

HEFRecHitGPU::HEFRecHitGPU(const edm::ParameterSet &ps)
    : uncalibRecHitCPUToken_{consumes<HGCUncalibratedRecHitCollection>(
          ps.getParameter<edm::InputTag>("HGCHEFUncalibRecHitsTok"))},
      recHitGPUToken_{produces<cms::cuda::Product<HGCRecHitGPUProduct>>()} {
  cdata_.keV2DIGI_ = ps.getParameter<double>("HGCHEF_keV2DIGI");
  cdata_.xmin_ = ps.getParameter<double>("minValSiPar");  //float
  cdata_.xmax_ = ps.getParameter<double>("maxValSiPar");  //float
  cdata_.aterm_ = ps.getParameter<double>("noiseSiPar");  //float
  cdata_.cterm_ = ps.getParameter<double>("constSiPar");  //float
  vdata_.fCPerMIP_ = ps.getParameter<std::vector<double>>("HGCHEF_fCPerMIP");
  vdata_.cce_ = ps.getParameter<edm::ParameterSet>("HGCHEF_cce").getParameter<std::vector<double>>("values");
  vdata_.noise_fC_ = ps.getParameter<edm::ParameterSet>("HGCHEF_noise_fC").getParameter<std::vector<double>>("values");
  vdata_.rcorr_ = ps.getParameter<std::vector<double>>("rcorr");
  vdata_.weights_ = ps.getParameter<std::vector<double>>("weights");
  cdata_.uncalib2GeV_ = 1e-6 / cdata_.keV2DIGI_;
  cdata_.layerOffset_ = 28;
  assert_sizes_constants_(vdata_);

  kcdata_ = new KernelConstantData<HGChefUncalibRecHitConstantData>(cdata_, vdata_);
  convert_constant_data_(kcdata_);

  tools_ = std::make_unique<hgcal::RecHitTools>();
}

HEFRecHitGPU::~HEFRecHitGPU() { delete kcdata_; }

std::string HEFRecHitGPU::assert_error_message_(std::string var, const size_t &s1, const size_t &s2) {
  std::string str1 = "The '";
  std::string str2 = "' array must be of size ";
  std::string str3 = " to hold the configuration data, but is of size ";
  return str1 + var + str2 + std::to_string(s1) + str3 + std::to_string(s2);
}

void HEFRecHitGPU::assert_sizes_constants_(const HGCConstantVectorData &vd) {
  if (vdata_.fCPerMIP_.size() != HGChefUncalibRecHitConstantData::hef_fCPerMIP)
    edm::LogError("WrongSize") << this->assert_error_message_(
        "fCPerMIP", HGChefUncalibRecHitConstantData::hef_fCPerMIP, vdata_.fCPerMIP_.size());
  else if (vdata_.cce_.size() != HGChefUncalibRecHitConstantData::hef_cce)
    edm::LogError("WrongSize") << this->assert_error_message_(
        "cce", HGChefUncalibRecHitConstantData::hef_cce, vdata_.cce_.size());
  else if (vdata_.noise_fC_.size() != HGChefUncalibRecHitConstantData::hef_noise_fC)
    edm::LogError("WrongSize") << this->assert_error_message_(
        "noise_fC", HGChefUncalibRecHitConstantData::hef_noise_fC, vdata_.noise_fC_.size());
  else if (vdata_.rcorr_.size() != HGChefUncalibRecHitConstantData::hef_rcorr)
    edm::LogError("WrongSize") << this->assert_error_message_(
        "rcorr", HGChefUncalibRecHitConstantData::hef_rcorr, vdata_.rcorr_.size());
  else if (vdata_.weights_.size() != HGChefUncalibRecHitConstantData::hef_weights)
    edm::LogError("WrongSize") << this->assert_error_message_(
        "weights", HGChefUncalibRecHitConstantData::hef_weights, vdata_.weights_.size());
}

void HEFRecHitGPU::beginRun(edm::Run const &, edm::EventSetup const &setup) {}

void HEFRecHitGPU::produce(edm::Event &event, const edm::EventSetup &setup) {
  cms::cuda::ScopedContextProduce ctx{event.streamID()};

  const auto &hits = event.get(uncalibRecHitCPUToken_);
  unsigned int nhits(hits.size());
  rechits_ = std::make_unique<HGCRecHitCollection>();

  if (nhits == 0)
    edm::LogError("HEFRecHitGPU") << "WARNING: no input hits!";

  prod_ = HGCRecHitGPUProduct(nhits, ctx.stream());
  d_uncalib_ = HGCUncalibRecHitDevice(nhits, ctx.stream());
  h_uncalib_ = HGCUncalibRecHitHost<HGChefUncalibratedRecHitCollection>(nhits, hits, ctx.stream());

  KernelManagerHGCalRecHit km(h_uncalib_.get(), d_uncalib_.get(), prod_.get());
  km.run_kernels(kcdata_, ctx.stream());

  ctx.emplace(event, recHitGPUToken_, std::move(prod_));
}

void HEFRecHitGPU::convert_constant_data_(KernelConstantData<HGChefUncalibRecHitConstantData> *kcdata) {
  for (size_t i = 0; i < kcdata->vdata_.fCPerMIP_.size(); ++i)
    kcdata->data_.fCPerMIP_[i] = kcdata->vdata_.fCPerMIP_[i];
  for (size_t i = 0; i < kcdata->vdata_.cce_.size(); ++i)
    kcdata->data_.cce_[i] = kcdata->vdata_.cce_[i];
  for (size_t i = 0; i < kcdata->vdata_.noise_fC_.size(); ++i)
    kcdata->data_.noise_fC_[i] = kcdata->vdata_.noise_fC_[i];
  for (size_t i = 0; i < kcdata->vdata_.rcorr_.size(); ++i)
    kcdata->data_.rcorr_[i] = kcdata->vdata_.rcorr_[i];
  for (size_t i = 0; i < kcdata->vdata_.weights_.size(); ++i)
    kcdata->data_.weights_[i] = kcdata->vdata_.weights_[i];
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HEFRecHitGPU);
