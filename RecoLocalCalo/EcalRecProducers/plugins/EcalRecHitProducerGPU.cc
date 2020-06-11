// framework
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

// format
#include "CUDADataFormats/EcalRecHitSoA/interface/EcalUncalibratedRecHit_soa.h"
#include "CUDADataFormats/EcalRecHitSoA/interface/EcalRecHit_soa.h"
#include "CUDADataFormats/EcalRecHitSoA/interface/RecoTypes.h"

// needed for definition of flags
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

// the kernels
#include "RecoLocalCalo/EcalRecAlgos/src/EcalRecHitBuilderKernels.h"

// conditions cpu
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRefRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAlphasRcd.h"
#include "CondFormats/DataRecord/interface/EcalLinearCorrectionsRcd.h"

// conditions gpu
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRechitADCToGeVConstantGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalIntercalibConstantsGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRechitChannelStatusGPU.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalLaserAPDPNRatiosGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalLaserAPDPNRatiosRefGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalLaserAlphasGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalLinearCorrectionsGPU.h"

#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

// configuration
#include "CommonTools/Utils/interface/StringToEnumValue.h"

class EcalRecHitProducerGPU : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit EcalRecHitProducerGPU(edm::ParameterSet const& ps);
  ~EcalRecHitProducerGPU() override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  using RecHitType = ecal::RecHit<ecal::Tag::soa>;
  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, edm::EventSetup const&) override;

private:
  // data
  uint32_t neb_, nee_;  // extremely important, in particular neb_

  // gpu input
  edm::EDGetTokenT<cms::cuda::Product<ecal::UncalibratedRecHit<ecal::Tag::ptr>>> uncalibRecHitsInEBToken_;
  edm::EDGetTokenT<cms::cuda::Product<ecal::UncalibratedRecHit<ecal::Tag::ptr>>> uncalibRecHitsInEEToken_;

  // event data
  ecal::rechit::EventOutputDataGPU eventOutputDataGPU_;

  cms::cuda::ContextState cudaState_;

  // gpu output
  edm::EDPutTokenT<cms::cuda::Product<ecal::RecHit<ecal::Tag::ptr>>> recHitsTokenEB_, recHitsTokenEE_;

  // configuration parameters
  ecal::rechit::ConfigurationParameters configParameters_;
  uint32_t maxNumberHits_;

  // conditions handles
  edm::ESHandle<EcalRechitADCToGeVConstantGPU> ADCToGeVConstantHandle_;
  edm::ESHandle<EcalIntercalibConstantsGPU> IntercalibConstantsHandle_;
  edm::ESHandle<EcalRechitChannelStatusGPU> ChannelStatusHandle_;

  edm::ESHandle<EcalLaserAPDPNRatiosGPU> LaserAPDPNRatiosHandle_;
  edm::ESHandle<EcalLaserAPDPNRatiosRefGPU> LaserAPDPNRatiosRefHandle_;
  edm::ESHandle<EcalLaserAlphasGPU> LaserAlphasHandle_;
  edm::ESHandle<EcalLinearCorrectionsGPU> LinearCorrectionsHandle_;

  // configuration
  std::vector<int> v_chstatus_;

  //
  // https://github.com/cms-sw/cmssw/blob/266e21cfc9eb409b093e4cf064f4c0a24c6ac293/RecoLocalCalo/EcalRecProducers/plugins/EcalRecHitWorkerSimple.h
  //

  // Associate reco flagbit ( outer vector) to many db status flags (inner vector)
  //   std::vector<std::vector<uint32_t> > v_DB_reco_flags_;
  std::vector<int>
      expanded_v_DB_reco_flags_;  // Transform a map in a vector      // FIXME AM: int or uint32 to be checked
  std::vector<uint32_t> expanded_Sizes_v_DB_reco_flags_;    // Saving the size for each piece
  std::vector<uint32_t> expanded_flagbit_v_DB_reco_flags_;  // And the "key" for each key

  uint32_t flagmask_;  // do not propagate channels with these flags on
};

void EcalRecHitProducerGPU::fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("uncalibrecHitsInLabelEB",
                          edm::InputTag("ecalUncalibRecHitProducerGPU", "EcalUncalibRecHitsEB"));
  desc.add<edm::InputTag>("uncalibrecHitsInLabelEE",
                          edm::InputTag("ecalUncalibRecHitProducerGPU", "EcalUncalibRecHitsEE"));

  desc.add<std::string>("recHitsLabelEB", "EcalRecHitsGPUEB");
  desc.add<std::string>("recHitsLabelEE", "EcalRecHitsGPUEE");

  desc.add<bool>("killDeadChannels", true);

  desc.add<double>("EBLaserMIN", 0.01);
  desc.add<double>("EELaserMIN", 0.01);
  desc.add<double>("EBLaserMAX", 30.0);
  desc.add<double>("EELaserMAX", 30.0);

  desc.add<uint32_t>("maxNumberHits", 20000);

  // ## db statuses to be exluded from reconstruction (some will be recovered)
  edm::ParameterSetDescription desc_ChannelStatusToBeExcluded;
  desc_ChannelStatusToBeExcluded.add<std::string>("kDAC");
  desc_ChannelStatusToBeExcluded.add<std::string>("kNoisy");
  desc_ChannelStatusToBeExcluded.add<std::string>("kNNoisy");
  desc_ChannelStatusToBeExcluded.add<std::string>("kFixedG6");
  desc_ChannelStatusToBeExcluded.add<std::string>("kFixedG1");
  desc_ChannelStatusToBeExcluded.add<std::string>("kFixedG0");
  desc_ChannelStatusToBeExcluded.add<std::string>("kNonRespondingIsolated");
  desc_ChannelStatusToBeExcluded.add<std::string>("kDeadVFE");
  desc_ChannelStatusToBeExcluded.add<std::string>("kDeadFE");
  desc_ChannelStatusToBeExcluded.add<std::string>("kNoDataNoTP");

  std::vector<edm::ParameterSet> default_ChannelStatusToBeExcluded(1);

  desc.addVPSet("ChannelStatusToBeExcluded", desc_ChannelStatusToBeExcluded, default_ChannelStatusToBeExcluded);
}

EcalRecHitProducerGPU::EcalRecHitProducerGPU(const edm::ParameterSet& ps) {
  //---- input
  uncalibRecHitsInEBToken_ = consumes<cms::cuda::Product<ecal::UncalibratedRecHit<ecal::Tag::ptr>>>(
      ps.getParameter<edm::InputTag>("uncalibrecHitsInLabelEB"));
  uncalibRecHitsInEEToken_ = consumes<cms::cuda::Product<ecal::UncalibratedRecHit<ecal::Tag::ptr>>>(
      ps.getParameter<edm::InputTag>("uncalibrecHitsInLabelEE"));

  //---- output
  recHitsTokenEB_ =
      produces<cms::cuda::Product<ecal::RecHit<ecal::Tag::ptr>>>(ps.getParameter<std::string>("recHitsLabelEB"));
  recHitsTokenEE_ =
      produces<cms::cuda::Product<ecal::RecHit<ecal::Tag::ptr>>>(ps.getParameter<std::string>("recHitsLabelEE"));

  //---- db statuses to be exluded from reconstruction
  v_chstatus_ = StringToEnumValue<EcalChannelStatusCode::Code>(
      ps.getParameter<std::vector<std::string>>("ChannelStatusToBeExcluded"));

  bool killDeadChannels = ps.getParameter<bool>("killDeadChannels");
  configParameters_.killDeadChannels = killDeadChannels;

  configParameters_.EBLaserMIN = ps.getParameter<double>("EBLaserMIN");
  configParameters_.EELaserMIN = ps.getParameter<double>("EELaserMIN");
  configParameters_.EBLaserMAX = ps.getParameter<double>("EBLaserMAX");
  configParameters_.EELaserMAX = ps.getParameter<double>("EELaserMAX");

  // max number of digis to allocate for
  maxNumberHits_ = ps.getParameter<uint32_t>("maxNumberHits");

  // allocate event output data
  eventOutputDataGPU_.allocate(configParameters_, maxNumberHits_);

  configParameters_.ChannelStatusToBeExcludedSize = v_chstatus_.size();

  
  // call CUDA API functions only if CUDA is available
  edm::Service<CUDAService> cs;
  if (cs and cs->enabled()) {
    
    cudaCheck(cudaMalloc((void**)&configParameters_.ChannelStatusToBeExcluded, sizeof(int) * v_chstatus_.size()));
    cudaCheck(cudaMemcpy(configParameters_.ChannelStatusToBeExcluded,
                         v_chstatus_.data(),
                         v_chstatus_.size() * sizeof(int),
                         cudaMemcpyHostToDevice));
  }
  
  //
  //     https://github.com/cms-sw/cmssw/blob/266e21cfc9eb409b093e4cf064f4c0a24c6ac293/RecoLocalCalo/EcalRecProducers/plugins/EcalRecHitWorkerSimple.cc
  //

  // Traslate string representation of flagsMapDBReco into enum values
  const edm::ParameterSet& p = ps.getParameter<edm::ParameterSet>("flagsMapDBReco");
  std::vector<std::string> recoflagbitsStrings = p.getParameterNames();
  //   v_DB_reco_flags_.resize(32);

  for (unsigned int i = 0; i != recoflagbitsStrings.size(); ++i) {
    EcalRecHit::Flags recoflagbit = (EcalRecHit::Flags)StringToEnumValue<EcalRecHit::Flags>(recoflagbitsStrings[i]);
    std::vector<std::string> dbstatus_s = p.getParameter<std::vector<std::string>>(recoflagbitsStrings[i]);
    //     std::vector<uint32_t> dbstatuses;
    for (unsigned int j = 0; j != dbstatus_s.size(); ++j) {
      EcalChannelStatusCode::Code dbstatus =
          (EcalChannelStatusCode::Code)StringToEnumValue<EcalChannelStatusCode::Code>(dbstatus_s[j]);
      //       dbstatuses.push_back(dbstatus);
      expanded_v_DB_reco_flags_.push_back(dbstatus);
    }

    expanded_Sizes_v_DB_reco_flags_.push_back(dbstatus_s.size());
    expanded_flagbit_v_DB_reco_flags_.push_back(recoflagbit);

    //     v_DB_reco_flags_[recoflagbit] = dbstatuses;
  }

  // call CUDA API functions only if CUDA is available
  if (cs and cs->enabled()) {
    // actual values
    cudaCheck(
        cudaMalloc((void**)&configParameters_.expanded_v_DB_reco_flags, sizeof(int) * expanded_v_DB_reco_flags_.size()));
    
    cudaCheck(cudaMemcpy(configParameters_.expanded_v_DB_reco_flags,
                         expanded_v_DB_reco_flags_.data(),
                         expanded_v_DB_reco_flags_.size() * sizeof(int),
                         cudaMemcpyHostToDevice));
    
    // sizes
    cudaCheck(cudaMalloc((void**)&configParameters_.expanded_Sizes_v_DB_reco_flags,
                         sizeof(uint32_t) * expanded_Sizes_v_DB_reco_flags_.size()));
    
    cudaCheck(cudaMemcpy(configParameters_.expanded_Sizes_v_DB_reco_flags,
                         expanded_Sizes_v_DB_reco_flags_.data(),
                         expanded_Sizes_v_DB_reco_flags_.size() * sizeof(uint32_t),
                         cudaMemcpyHostToDevice));
    
    // keys
    cudaCheck(cudaMalloc((void**)&configParameters_.expanded_flagbit_v_DB_reco_flags,
                         sizeof(uint32_t) * expanded_flagbit_v_DB_reco_flags_.size()));
    
    cudaCheck(cudaMemcpy(configParameters_.expanded_flagbit_v_DB_reco_flags,
                         expanded_flagbit_v_DB_reco_flags_.data(),
                         expanded_flagbit_v_DB_reco_flags_.size() * sizeof(uint32_t),
                         cudaMemcpyHostToDevice));
  }
  
  configParameters_.expanded_v_DB_reco_flagsSize = expanded_flagbit_v_DB_reco_flags_.size();

  flagmask_ = 0;
  flagmask_ |= 0x1 << EcalRecHit::kNeighboursRecovered;
  flagmask_ |= 0x1 << EcalRecHit::kTowerRecovered;
  flagmask_ |= 0x1 << EcalRecHit::kDead;
  flagmask_ |= 0x1 << EcalRecHit::kKilled;
  flagmask_ |= 0x1 << EcalRecHit::kTPSaturated;
  flagmask_ |= 0x1 << EcalRecHit::kL1SpikeFlag;

  configParameters_.flagmask = flagmask_;

  // for recovery and killing

  configParameters_.recoverEBIsolatedChannels = ps.getParameter<bool>("recoverEBIsolatedChannels");
  configParameters_.recoverEEIsolatedChannels = ps.getParameter<bool>("recoverEEIsolatedChannels");
  configParameters_.recoverEBVFE = ps.getParameter<bool>("recoverEBVFE");
  configParameters_.recoverEEVFE = ps.getParameter<bool>("recoverEEVFE");
  configParameters_.recoverEBFE = ps.getParameter<bool>("recoverEBFE");
  configParameters_.recoverEEFE = ps.getParameter<bool>("recoverEEFE");
}

EcalRecHitProducerGPU::~EcalRecHitProducerGPU() {
  
  edm::Service<CUDAService> cs;
  if (cs and cs->enabled()) {
    // free event ouput data
    eventOutputDataGPU_.deallocate(configParameters_);

    // FIXME AM: do I need to do this?
    //           Or can I do it as part of "deallocate" ?
    cudaCheck(cudaFree(configParameters_.ChannelStatusToBeExcluded));

    cudaCheck(cudaFree(configParameters_.expanded_v_DB_reco_flags));
    cudaCheck(cudaFree(configParameters_.expanded_Sizes_v_DB_reco_flags));
    cudaCheck(cudaFree(configParameters_.expanded_flagbit_v_DB_reco_flags));
  }
}

void EcalRecHitProducerGPU::acquire(edm::Event const& event,
                                    edm::EventSetup const& setup,
                                    edm::WaitingTaskWithArenaHolder holder) {
  // cuda products
  auto const& ebUncalibRecHitsProduct = event.get(uncalibRecHitsInEBToken_);
  auto const& eeUncalibRecHitsProduct = event.get(uncalibRecHitsInEEToken_);
  // raii
  cms::cuda::ScopedContextAcquire ctx{ebUncalibRecHitsProduct, std::move(holder), cudaState_};
  // get actual object
  auto const& ebUncalibRecHits = ctx.get(ebUncalibRecHitsProduct);
  auto const& eeUncalibRecHits = ctx.get(eeUncalibRecHitsProduct);

  ecal::rechit::EventInputDataGPU inputDataGPU{ebUncalibRecHits, eeUncalibRecHits};

  neb_ = ebUncalibRecHits.size;
  nee_ = eeUncalibRecHits.size;
  // std::cout << " [EcalRecHitProducerGPU::acquire]  neb_:nee_ = " << neb_ << " : " << nee_ << std::endl;

  if ((neb_ + nee_) > maxNumberHits_) {
    edm::LogError("EcalRecHitProducerGPU") << "max number of channels exceeded. See options 'maxNumberHits' ";
  }
  
  int nchannelsEB = ebUncalibRecHits.size;  // --> offsetForInput, first EB and then EE

  // conditions
  // - laser correction
  // - IC
  // - adt2gev

  //
  setup.get<EcalADCToGeVConstantRcd>().get(ADCToGeVConstantHandle_);
  setup.get<EcalIntercalibConstantsRcd>().get(IntercalibConstantsHandle_);
  setup.get<EcalChannelStatusRcd>().get(ChannelStatusHandle_);

  setup.get<EcalLaserAPDPNRatiosRcd>().get(LaserAPDPNRatiosHandle_);
  setup.get<EcalLaserAPDPNRatiosRefRcd>().get(LaserAPDPNRatiosRefHandle_);
  setup.get<EcalLaserAlphasRcd>().get(LaserAlphasHandle_);
  setup.get<EcalLinearCorrectionsRcd>().get(LinearCorrectionsHandle_);

  auto const& ADCToGeVConstantProduct = ADCToGeVConstantHandle_->getProduct(ctx.stream());
  auto const& IntercalibConstantsProduct = IntercalibConstantsHandle_->getProduct(ctx.stream());
  auto const& ChannelStatusProduct = ChannelStatusHandle_->getProduct(ctx.stream());

  auto const& LaserAPDPNRatiosProduct = LaserAPDPNRatiosHandle_->getProduct(ctx.stream());
  auto const& LaserAPDPNRatiosRefProduct = LaserAPDPNRatiosRefHandle_->getProduct(ctx.stream());
  auto const& LaserAlphasProduct = LaserAlphasHandle_->getProduct(ctx.stream());
  auto const& LinearCorrectionsProduct = LinearCorrectionsHandle_->getProduct(ctx.stream());

  // bundle up conditions
  ecal::rechit::ConditionsProducts conditions{ADCToGeVConstantProduct,
                                              IntercalibConstantsProduct,
                                              ChannelStatusProduct,
                                              //
                                              LaserAPDPNRatiosProduct,
                                              LaserAPDPNRatiosRefProduct,
                                              LaserAlphasProduct,
                                              LinearCorrectionsProduct,
                                              //
                                              IntercalibConstantsHandle_->getOffset()};

  //
  // schedule algorithms
  //

  edm::TimeValue_t event_time = event.time().value();

  ecal::rechit::create_ecal_rehit(inputDataGPU,
                                  eventOutputDataGPU_,
                                  //     eventDataForScratchGPU_,
                                  conditions,
                                  configParameters_,
                                  nchannelsEB,
                                  event_time,
                                  ctx.stream());

  cudaCheck(cudaGetLastError());
}

void EcalRecHitProducerGPU::produce(edm::Event& event, edm::EventSetup const& setup) {
  //DurationMeasurer<std::chrono::milliseconds> timer{std::string{"produce duration"}};
  cms::cuda::ScopedContextProduce ctx{cudaState_};

  // copy construct output collections
  // note, output collections do not own device memory!
  ecal::RecHit<ecal::Tag::ptr> ebRecHits{eventOutputDataGPU_};
  ecal::RecHit<ecal::Tag::ptr> eeRecHits{eventOutputDataGPU_};

  // set the size of eb and ee
  ebRecHits.size = neb_;
  eeRecHits.size = nee_;

  // shift ptrs for ee
  eeRecHits.energy += neb_;
  eeRecHits.chi2 += neb_;
  eeRecHits.did += neb_;
  eeRecHits.time += neb_;
  eeRecHits.extra += neb_;
  eeRecHits.flagBits += neb_;

  // put into the event
  ctx.emplace(event, recHitsTokenEB_, std::move(ebRecHits));
  ctx.emplace(event, recHitsTokenEE_, std::move(eeRecHits));
}

DEFINE_FWK_MODULE(EcalRecHitProducerGPU);
