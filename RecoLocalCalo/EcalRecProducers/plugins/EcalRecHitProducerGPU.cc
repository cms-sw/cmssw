#include "CUDADataFormats/EcalRecHitSoA/interface/EcalRecHit.h"
#include "CUDADataFormats/EcalRecHitSoA/interface/EcalUncalibratedRecHit.h"
#include "CUDADataFormats/EcalRecHitSoA/interface/RecoTypes.h"
#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRefRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAlphasRcd.h"
#include "CondFormats/DataRecord/interface/EcalLinearCorrectionsRcd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDACore/interface/JobConfigurationGPURecord.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalIntercalibConstantsGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalLaserAPDPNRatiosGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalLaserAPDPNRatiosRefGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalLaserAlphasGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalLinearCorrectionsGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRecHitParametersGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRechitADCToGeVConstantGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRechitChannelStatusGPU.h"

#include "EcalRecHitBuilderKernels.h"

class EcalRecHitProducerGPU : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit EcalRecHitProducerGPU(edm::ParameterSet const& ps);
  ~EcalRecHitProducerGPU() override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, edm::EventSetup const&) override;

private:
  // data
  uint32_t neb_, nee_;  // extremely important, in particular neb_

  // gpu input
  using InputProduct = cms::cuda::Product<ecal::UncalibratedRecHit<calo::common::DevStoragePolicy>>;
  edm::EDGetTokenT<InputProduct> uncalibRecHitsInEBToken_;
  edm::EDGetTokenT<InputProduct> uncalibRecHitsInEEToken_;

  // event data
  ecal::rechit::EventOutputDataGPU eventOutputDataGPU_;

  cms::cuda::ContextState cudaState_;

  // gpu output
  using OutputProduct = cms::cuda::Product<ecal::RecHit<calo::common::DevStoragePolicy>>;
  edm::EDPutTokenT<OutputProduct> recHitsTokenEB_, recHitsTokenEE_;

  // configuration parameters
  ecal::rechit::ConfigurationParameters configParameters_;

  // conditions handles
  edm::ESHandle<EcalRechitADCToGeVConstantGPU> ADCToGeVConstantHandle_;
  edm::ESHandle<EcalIntercalibConstantsGPU> IntercalibConstantsHandle_;
  edm::ESHandle<EcalRechitChannelStatusGPU> ChannelStatusHandle_;

  edm::ESHandle<EcalLaserAPDPNRatiosGPU> LaserAPDPNRatiosHandle_;
  edm::ESHandle<EcalLaserAPDPNRatiosRefGPU> LaserAPDPNRatiosRefHandle_;
  edm::ESHandle<EcalLaserAlphasGPU> LaserAlphasHandle_;
  edm::ESHandle<EcalLinearCorrectionsGPU> LinearCorrectionsHandle_;
  edm::ESHandle<EcalRecHitParametersGPU> recHitParametersHandle_;

  // Associate reco flagbit (outer vector) to many db status flags (inner vector)
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

  desc.add<uint32_t>("maxNumberHitsEB", 61200);
  desc.add<uint32_t>("maxNumberHitsEE", 14648);
}

EcalRecHitProducerGPU::EcalRecHitProducerGPU(const edm::ParameterSet& ps) {
  //---- input
  uncalibRecHitsInEBToken_ = consumes<InputProduct>(ps.getParameter<edm::InputTag>("uncalibrecHitsInLabelEB"));
  uncalibRecHitsInEEToken_ = consumes<InputProduct>(ps.getParameter<edm::InputTag>("uncalibrecHitsInLabelEE"));

  //---- output
  recHitsTokenEB_ = produces<OutputProduct>(ps.getParameter<std::string>("recHitsLabelEB"));
  recHitsTokenEE_ = produces<OutputProduct>(ps.getParameter<std::string>("recHitsLabelEE"));

  bool killDeadChannels = ps.getParameter<bool>("killDeadChannels");
  configParameters_.killDeadChannels = killDeadChannels;

  configParameters_.EBLaserMIN = ps.getParameter<double>("EBLaserMIN");
  configParameters_.EELaserMIN = ps.getParameter<double>("EELaserMIN");
  configParameters_.EBLaserMAX = ps.getParameter<double>("EBLaserMAX");
  configParameters_.EELaserMAX = ps.getParameter<double>("EELaserMAX");

  // max number of digis to allocate for
  configParameters_.maxNumberHitsEB = ps.getParameter<uint32_t>("maxNumberHitsEB");
  configParameters_.maxNumberHitsEE = ps.getParameter<uint32_t>("maxNumberHitsEE");

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

EcalRecHitProducerGPU::~EcalRecHitProducerGPU() {}

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

  if ((neb_ > configParameters_.maxNumberHitsEB) || (nee_ > configParameters_.maxNumberHitsEE)) {
    edm::LogError("EcalRecHitProducerGPU")
        << "max number of channels exceeded. See options 'maxNumberHitsEB and maxNumberHitsEE' ";
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
  setup.get<JobConfigurationGPURecord>().get(recHitParametersHandle_);

  auto const& ADCToGeVConstantProduct = ADCToGeVConstantHandle_->getProduct(ctx.stream());
  auto const& IntercalibConstantsProduct = IntercalibConstantsHandle_->getProduct(ctx.stream());
  auto const& ChannelStatusProduct = ChannelStatusHandle_->getProduct(ctx.stream());

  auto const& LaserAPDPNRatiosProduct = LaserAPDPNRatiosHandle_->getProduct(ctx.stream());
  auto const& LaserAPDPNRatiosRefProduct = LaserAPDPNRatiosRefHandle_->getProduct(ctx.stream());
  auto const& LaserAlphasProduct = LaserAlphasHandle_->getProduct(ctx.stream());
  auto const& LinearCorrectionsProduct = LinearCorrectionsHandle_->getProduct(ctx.stream());
  auto const& recHitParametersProduct = recHitParametersHandle_->getProduct(ctx.stream());

  // set config ptrs : this is done to avoid changing things downstream
  configParameters_.ChannelStatusToBeExcluded = recHitParametersProduct.ChannelStatusToBeExcluded;
  configParameters_.ChannelStatusToBeExcludedSize = std::get<0>(recHitParametersHandle_->getValues()).get().size();
  configParameters_.expanded_v_DB_reco_flags = recHitParametersProduct.expanded_v_DB_reco_flags;
  configParameters_.expanded_Sizes_v_DB_reco_flags = recHitParametersProduct.expanded_Sizes_v_DB_reco_flags;
  configParameters_.expanded_flagbit_v_DB_reco_flags = recHitParametersProduct.expanded_flagbit_v_DB_reco_flags;
  configParameters_.expanded_v_DB_reco_flagsSize = std::get<3>(recHitParametersHandle_->getValues()).get().size();

  // bundle up conditions
  ecal::rechit::ConditionsProducts conditions{ADCToGeVConstantProduct,
                                              IntercalibConstantsProduct,
                                              ChannelStatusProduct,
                                              LaserAPDPNRatiosProduct,
                                              LaserAPDPNRatiosRefProduct,
                                              LaserAlphasProduct,
                                              LinearCorrectionsProduct,
                                              IntercalibConstantsHandle_->getOffset()};

  // dev mem
  eventOutputDataGPU_.allocate(configParameters_, ctx.stream());

  //
  // schedule algorithms
  //

  edm::TimeValue_t event_time = event.time().value();

  ecal::rechit::create_ecal_rehit(
      inputDataGPU, eventOutputDataGPU_, conditions, configParameters_, nchannelsEB, event_time, ctx.stream());

  cudaCheck(cudaGetLastError());
}

void EcalRecHitProducerGPU::produce(edm::Event& event, edm::EventSetup const& setup) {
  cms::cuda::ScopedContextProduce ctx{cudaState_};

  eventOutputDataGPU_.recHitsEB.size = neb_;
  eventOutputDataGPU_.recHitsEE.size = nee_;

  // put into the event
  ctx.emplace(event, recHitsTokenEB_, std::move(eventOutputDataGPU_.recHitsEB));
  ctx.emplace(event, recHitsTokenEE_, std::move(eventOutputDataGPU_.recHitsEE));
}

DEFINE_FWK_MODULE(EcalRecHitProducerGPU);
