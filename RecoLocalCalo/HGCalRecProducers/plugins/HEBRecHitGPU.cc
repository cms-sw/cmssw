#include "HEBRecHitGPU.h"

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

  h_uncalibSoA_ = new HGCUncalibratedRecHitSoA();
  d_uncalibSoA_ = new HGCUncalibratedRecHitSoA();
  d_calibSoA_ = new HGCRecHitSoA();
  kcdata_ = new KernelConstantData<HGChebUncalibratedRecHitConstantData>(cdata_, vdata_);

  tools_ = std::make_unique<hgcal::RecHitTools>();
}

HEBRecHitGPU::~HEBRecHitGPU() {
  delete kcdata_;
  delete h_uncalibSoA_;
  delete d_uncalibSoA_;
  delete d_calibSoA_;
}

std::string HEBRecHitGPU::assert_error_message_(std::string var, const size_t& s) {
  std::string str1 = "The '";
  std::string str2 = "' array must be at least of size ";
  std::string str3 = " to hold the configuration data.";
  return str1 + var + str2 + std::to_string(s) + str3;
}

void HEBRecHitGPU::assert_sizes_constants_(const HGCConstantVectorData& vd) {
  if (vdata_.weights_.size() > HGChebUncalibratedRecHitConstantData::heb_weights)
    cms::cuda::LogError("MaxSizeExceeded") << this->assert_error_message_("weights", vdata_.fCPerMIP_.size());
}

void HEBRecHitGPU::beginRun(edm::Run const&, edm::EventSetup const& setup) {}

void HEBRecHitGPU::acquire(edm::Event const& event, edm::EventSetup const& setup, edm::WaitingTaskWithArenaHolder w) {
  const cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(w), ctxState_};

  event.getByToken(uncalibRecHitCPUToken_, handle_);
  const auto& hits = *handle_;

  unsigned int nhits = hits.size();
  rechits_ = std::make_unique<HGCRecHitCollection>();

  if (nhits == 0)
    cms::cuda::LogError("HEBRecHitGPU") << "WARNING: no input hits!";

  prod_ = HGCRecHitGPUProduct(nhits, ctx.stream());
  allocate_memory_(ctx.stream());
  convert_constant_data_(kcdata_);
  convert_collection_data_to_soa_(nhits, hits, h_uncalibSoA_);

  KernelManagerHGCalRecHit km(h_uncalibSoA_, d_uncalibSoA_, d_calibSoA_);
  km.run_kernels(kcdata_, ctx.stream());
}

void HEBRecHitGPU::produce(edm::Event& event, const edm::EventSetup& setup) {
  cms::cuda::ScopedContextProduce ctx{ctxState_};
  ctx.emplace(event, recHitGPUToken_, std::move(prod_));
}

void HEBRecHitGPU::allocate_memory_(const cudaStream_t& stream) {
  //_allocate memory for uncalibrated hits on the host
  memory::allocation::uncalibRecHitHost(prod_.nHits(), prod_.stride(), h_uncalibSoA_, stream);
  //_allocate memory for uncalibrated hits on the device
  memory::allocation::uncalibRecHitDevice(prod_.nHits(), prod_.stride(), d_uncalibSoA_, stream);
  //point SoA to allocated memory for calibrated hits on the device
  memory::allocation::calibRecHitDevice(prod_.nHits(), prod_.stride(), prod_.nBytes(), d_calibSoA_, prod_.get());
}

void HEBRecHitGPU::convert_constant_data_(KernelConstantData<HGChebUncalibratedRecHitConstantData>* kcdata) {
  for (size_t i = 0; i < kcdata->vdata_.weights_.size(); ++i)
    kcdata->data_.weights_[i] = kcdata->vdata_.weights_[i];
}

void HEBRecHitGPU::convert_collection_data_to_soa_(const uint32_t& nhits,
                                                   const HGChebUncalibratedRecHitCollection& coll,
                                                   HGCUncalibratedRecHitSoA* soa) {
  for (unsigned i = 0; i < nhits; ++i) {
    soa->amplitude_[i] = coll[i].amplitude();
    soa->pedestal_[i] = coll[i].pedestal();
    soa->jitter_[i] = coll[i].jitter();
    soa->chi2_[i] = coll[i].chi2();
    soa->OOTamplitude_[i] = coll[i].outOfTimeEnergy();
    soa->OOTchi2_[i] = coll[i].outOfTimeChi2();
    soa->flags_[i] = coll[i].flags();
    soa->aux_[i] = 0;
    soa->id_[i] = coll[i].id().rawId();
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HEBRecHitGPU);
