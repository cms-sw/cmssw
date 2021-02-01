#include <iostream>
#include <string>
#include <memory>
#include <chrono>
#include <cuda_runtime.h>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"

#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/ContextState.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "HeterogeneousHGCalProducerMemoryWrapper.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoLocalCalo/HGCalRecProducers/plugins/KernelManagerHGCalRecHit.h"

#include "CUDADataFormats/HGCal/interface/HGCRecHitGPUProduct.h"

class EERecHitGPU : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit EERecHitGPU(const edm::ParameterSet &ps);
  ~EERecHitGPU() override;
  void beginRun(edm::Run const &, edm::EventSetup const &) override;

  void acquire(edm::Event const &, edm::EventSetup const &, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event &, const edm::EventSetup &) override;

private:
  edm::EDGetTokenT<HGCeeUncalibratedRecHitCollection> uncalibRecHitCPUToken_;
  edm::EDPutTokenT<cms::cuda::Product<HGCRecHitGPUProduct>> recHitGPUToken_;

  edm::Handle<HGCeeUncalibratedRecHitCollection> handle_;
  std::unique_ptr<HGCeeRecHitCollection> rechits_;
  cms::cuda::ContextState ctxState_;

  //constants
  HGCeeUncalibratedRecHitConstantData cdata_;
  HGCConstantVectorData vdata_;

  //memory
  std::string assert_error_message_(std::string, const size_t &, const size_t &);
  void assert_sizes_constants_(const HGCConstantVectorData &);
  void allocate_memory_(const cudaStream_t &);

  //conditions (geometry, topology, ...)
  std::unique_ptr<hgcal::RecHitTools> tools_;

  //data processing
  void convert_collection_data_to_soa_(const uint32_t &,
                                       const HGCeeUncalibratedRecHitCollection &);
  void convert_constant_data_(KernelConstantData<HGCeeUncalibratedRecHitConstantData> *);

  HGCRecHitGPUProduct prod_;
  HGCUncalibratedRecHitSoA h_uncalibSoA_, d_uncalibSoA_;
  HGCRecHitSoA d_calibSoA_;

  KernelConstantData<HGCeeUncalibratedRecHitConstantData> *kcdata_;
};

EERecHitGPU::EERecHitGPU(const edm::ParameterSet& ps)
    : uncalibRecHitCPUToken_{consumes<HGCUncalibratedRecHitCollection>(
          ps.getParameter<edm::InputTag>("HGCEEUncalibRecHitsTok"))},
      recHitGPUToken_{produces<cms::cuda::Product<HGCRecHitGPUProduct>>()} {
  cdata_.keV2DIGI_ = ps.getParameter<double>("HGCEE_keV2DIGI");
  cdata_.xmin_ = ps.getParameter<double>("minValSiPar");  //float
  cdata_.xmax_ = ps.getParameter<double>("maxValSiPar");  //float
  cdata_.aterm_ = ps.getParameter<double>("noiseSiPar");  //float
  cdata_.cterm_ = ps.getParameter<double>("constSiPar");  //float
  vdata_.fCPerMIP_ = ps.getParameter<std::vector<double>>("HGCEE_fCPerMIP");
  vdata_.cce_ = ps.getParameter<edm::ParameterSet>("HGCEE_cce").getParameter<std::vector<double>>("values");
  vdata_.noise_fC_ = ps.getParameter<edm::ParameterSet>("HGCEE_noise_fC").getParameter<std::vector<double>>("values");
  vdata_.rcorr_ = ps.getParameter<std::vector<double>>("rcorr");
  vdata_.weights_ = ps.getParameter<std::vector<double>>("weights");
  cdata_.uncalib2GeV_ = 1e-6 / cdata_.keV2DIGI_;
  assert_sizes_constants_(vdata_);

  kcdata_ = new KernelConstantData<HGCeeUncalibratedRecHitConstantData>(cdata_, vdata_);

  tools_ = std::make_unique<hgcal::RecHitTools>();
}

EERecHitGPU::~EERecHitGPU() {
  delete kcdata_;
}

std::string EERecHitGPU::assert_error_message_(std::string var, const size_t& s1, const size_t& s2) {
  std::string str1 = "The '";
  std::string str2 = "' array must be at least of size ";
  std::string str3 = " to hold the configuration data, but is of size ";
  return str1 + var + str2 + std::to_string(s1) + str3 + std::to_string(s2);
}

void EERecHitGPU::assert_sizes_constants_(const HGCConstantVectorData& vd) {
  if (vdata_.fCPerMIP_.size() < HGCeeUncalibratedRecHitConstantData::ee_fCPerMIP)
    edm::LogError("WrongSize") << this->assert_error_message_(
        "fCPerMIP", HGCeeUncalibratedRecHitConstantData::ee_fCPerMIP, vdata_.fCPerMIP_.size());
  else if (vdata_.cce_.size() < HGCeeUncalibratedRecHitConstantData::ee_cce)
    edm::LogError("WrongSize") << this->assert_error_message_(
        "cce", HGCeeUncalibratedRecHitConstantData::ee_cce, vdata_.cce_.size());
  else if (vdata_.noise_fC_.size() < HGCeeUncalibratedRecHitConstantData::ee_noise_fC)
    edm::LogError("WrongSize") << this->assert_error_message_(
        "noise_fC", HGCeeUncalibratedRecHitConstantData::ee_noise_fC, vdata_.noise_fC_.size());
  else if (vdata_.rcorr_.size() < HGCeeUncalibratedRecHitConstantData::ee_rcorr)
    edm::LogError("WrongSize") << this->assert_error_message_(
        "rcorr", HGCeeUncalibratedRecHitConstantData::ee_rcorr, vdata_.rcorr_.size());
  else if (vdata_.weights_.size() < HGCeeUncalibratedRecHitConstantData::ee_weights)
    edm::LogError("WrongSize") << this->assert_error_message_(
        "weights", HGCeeUncalibratedRecHitConstantData::ee_weights, vdata_.weights_.size());
}

void EERecHitGPU::beginRun(edm::Run const&, edm::EventSetup const& setup) {}

void EERecHitGPU::acquire(edm::Event const& event, edm::EventSetup const& setup, edm::WaitingTaskWithArenaHolder w) {
  const cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(w), ctxState_};

  event.getByToken(uncalibRecHitCPUToken_, handle_);
  const auto& hits = *handle_;

  unsigned int nhits = hits.size();
  rechits_ = std::make_unique<HGCRecHitCollection>();

  if (nhits == 0)
    cms::cuda::LogError("EERecHitGPU") << "WARNING: no input hits!";

  prod_ = HGCRecHitGPUProduct(nhits, ctx.stream());
  allocate_memory_(ctx.stream());
  convert_constant_data_(kcdata_);
  convert_collection_data_to_soa_(nhits, hits);

  KernelManagerHGCalRecHit km(h_uncalibSoA_, d_uncalibSoA_, d_calibSoA_);
  km.run_kernels(kcdata_, ctx.stream());
}

void EERecHitGPU::produce(edm::Event& event, const edm::EventSetup& setup) {
  cms::cuda::ScopedContextProduce ctx{ctxState_};
  ctx.emplace(event, recHitGPUToken_, std::move(prod_));
}

void EERecHitGPU::allocate_memory_(const cudaStream_t& stream) {
  //_allocate memory for uncalibrated hits on the host
  memory::allocation::uncalibRecHitHost(prod_.nHits(), prod_.stride(), h_uncalibSoA_, stream);
  //_allocate memory for uncalibrated hits on the device
  memory::allocation::uncalibRecHitDevice(prod_.nHits(), prod_.stride(), d_uncalibSoA_, stream);
  //point SoA to allocated memory for calibrated hits on the device
  memory::allocation::calibRecHitDevice(prod_.nHits(), prod_.stride(), prod_.nBytes(), d_calibSoA_, prod_.get());
}

void EERecHitGPU::convert_constant_data_(KernelConstantData<HGCeeUncalibratedRecHitConstantData>* kcdata) {
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

void EERecHitGPU::convert_collection_data_to_soa_(const uint32_t& nhits,
                                                  const HGCeeUncalibratedRecHitCollection& coll) {
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
DEFINE_FWK_MODULE(EERecHitGPU);
