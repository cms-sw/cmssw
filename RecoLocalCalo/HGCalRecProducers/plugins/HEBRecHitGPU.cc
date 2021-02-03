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

#include "HeterogeneousHGCalProducerMemoryWrapper.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/KernelManagerHGCalRecHit.h"
#include "CUDADataFormats/HGCal/interface/HGCRecHitGPUProduct.h"

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
  HGChebUncalibratedRecHitConstantData cdata_;
  HGCConstantVectorData vdata_;

  //memory
  std::string assert_error_message_(std::string, const size_t &);
  void assert_sizes_constants_(const HGCConstantVectorData &);

  //conditions (geometry, topology, ...)
  std::unique_ptr<hgcal::RecHitTools> tools_;

  //data processing
  void convert_collection_data_to_soa_(const uint32_t &,
                                       const HGChebUncalibratedRecHitCollection &);
  void convert_constant_data_(KernelConstantData<HGChebUncalibratedRecHitConstantData> *);

  HGCRecHitGPUProduct prod_;
  HGCUncalibratedRecHitSoA h_uncalibSoA_, d_uncalibSoA_;
  HGCRecHitSoA d_calibSoA_;

  KernelConstantData<HGChebUncalibratedRecHitConstantData> *kcdata_;
};

HEBRecHitGPU::HEBRecHitGPU(const edm::ParameterSet& ps)
    : uncalibRecHitCPUToken_{consumes<HGCUncalibratedRecHitCollection>(
          ps.getParameter<edm::InputTag>("HGCHEBUncalibRecHitsTok"))},
      recHitGPUToken_{produces<cms::cuda::Product<HGCRecHitGPUProduct>>()} {
  cdata_.keV2DIGI_ = ps.getParameter<double>("HGCHEB_keV2DIGI");
  cdata_.noise_MIP_ = ps.getParameter<edm::ParameterSet>("HGCHEB_noise_MIP").getParameter<double>("noise_MIP");
  vdata_.weights_ = ps.getParameter<std::vector<double>>("weights");
  cdata_.uncalib2GeV_ = 1e-6 / cdata_.keV2DIGI_;
  cdata_.layerOffset_ = 28;
  assert_sizes_constants_(vdata_);

  kcdata_ = new KernelConstantData<HGChebUncalibratedRecHitConstantData>(cdata_, vdata_);

  tools_ = std::make_unique<hgcal::RecHitTools>();
}

HEBRecHitGPU::~HEBRecHitGPU() {
  delete kcdata_;
}

std::string HEBRecHitGPU::assert_error_message_(std::string var, const size_t& s) {
  std::string str1 = "The '";
  std::string str2 = "' array must be at least of size ";
  std::string str3 = " to hold the configuration data.";
  return str1 + var + str2 + std::to_string(s) + str3;
}

void HEBRecHitGPU::assert_sizes_constants_(const HGCConstantVectorData& vd) {
  if (vdata_.weights_.size() > HGChebUncalibratedRecHitConstantData::heb_weights)
    edm::LogError("MaxSizeExceeded") << this->assert_error_message_("weights", vdata_.fCPerMIP_.size());
}

void HEBRecHitGPU::beginRun(edm::Run const&, edm::EventSetup const& setup) {}

void HEBRecHitGPU::produce(edm::Event& event, const edm::EventSetup& setup) {
  cms::cuda::ScopedContextProduce ctx{event.streamID()};

  const auto& hits = event.get(uncalibRecHitCPUToken_);
  unsigned int nhits = hits.size();
  rechits_ = std::make_unique<HGCRecHitCollection>();

  if (nhits == 0)
    cms::cuda::LogError("HEBRecHitGPU") << "WARNING: no input hits!";

  prod_ = HGCRecHitGPUProduct(nhits, ctx.stream());
  
  //_allocate memory for uncalibrated hits on the host
  cms::cuda::host::unique_ptr<std::byte[]> dummy_hmem = memory::allocation::uncalibRecHitHost(prod_.nHits(), prod_.stride(), h_uncalibSoA_, ctx.stream());
  //_allocate memory for uncalibrated hits on the device
  cms::cuda::device::unique_ptr<std::byte[]> dummy_dmem = memory::allocation::uncalibRecHitDevice(prod_.nHits(), prod_.stride(), d_uncalibSoA_, ctx.stream());
  //point SoA to allocated memory for calibrated hits on the device
  memory::allocation::calibRecHitDevice(prod_.nHits(), prod_.stride(), prod_.nBytes(), d_calibSoA_, prod_.get());
  
  convert_constant_data_(kcdata_);
  convert_collection_data_to_soa_(nhits, hits);

  KernelManagerHGCalRecHit km(h_uncalibSoA_, d_uncalibSoA_, d_calibSoA_);
  km.run_kernels(kcdata_, ctx.stream());
  
  ctx.emplace(event, recHitGPUToken_, std::move(prod_));
}

void HEBRecHitGPU::convert_constant_data_(KernelConstantData<HGChebUncalibratedRecHitConstantData>* kcdata) {
  for (size_t i = 0; i < kcdata->vdata_.weights_.size(); ++i)
    kcdata->data_.weights_[i] = kcdata->vdata_.weights_[i];
}

void HEBRecHitGPU::convert_collection_data_to_soa_(const uint32_t& nhits,
                                                   const HGChebUncalibratedRecHitCollection& coll) {
  for (unsigned i = 0; i < nhits; ++i) {
    h_uncalibSoA_.amplitude_[i] = coll[i].amplitude();
    h_uncalibSoA_.pedestal_[i] = coll[i].pedestal();
    h_uncalibSoA_.jitter_[i] = coll[i].jitter();
    h_uncalibSoA_.chi2_[i] = coll[i].chi2();
    h_uncalibSoA_.OOTamplitude_[i] = coll[i].outOfTimeEnergy();
    h_uncalibSoA_.OOTchi2_[i] = coll[i].outOfTimeChi2();
    h_uncalibSoA_.flags_[i] = coll[i].flags();
    h_uncalibSoA_.aux_[i] = 0;
    h_uncalibSoA_.id_[i] = coll[i].id().rawId();
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HEBRecHitGPU);
