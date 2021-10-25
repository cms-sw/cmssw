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
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsGPU.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosGPU.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosRefGPU.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAlphasGPU.h"
#include "CondFormats/EcalObjects/interface/EcalLinearCorrectionsGPU.h"
#include "CondFormats/EcalObjects/interface/EcalRecHitParametersGPU.h"
#include "CondFormats/EcalObjects/interface/EcalRechitADCToGeVConstantGPU.h"
#include "CondFormats/EcalObjects/interface/EcalRechitChannelStatusGPU.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "HeterogeneousCore/CUDACore/interface/JobConfigurationGPURecord.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

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

  // conditions tokens
  edm::ESGetToken<EcalRechitADCToGeVConstantGPU, EcalADCToGeVConstantRcd> tokenADCToGeVConstant_;
  edm::ESGetToken<EcalIntercalibConstantsGPU, EcalIntercalibConstantsRcd> tokenIntercalibConstants_;
  edm::ESGetToken<EcalRechitChannelStatusGPU, EcalChannelStatusRcd> tokenChannelStatus_;
  edm::ESGetToken<EcalLaserAPDPNRatiosGPU, EcalLaserAPDPNRatiosRcd> tokenLaserAPDPNRatios_;
  edm::ESGetToken<EcalLaserAPDPNRatiosRefGPU, EcalLaserAPDPNRatiosRefRcd> tokenLaserAPDPNRatiosRef_;
  edm::ESGetToken<EcalLaserAlphasGPU, EcalLaserAlphasRcd> tokenLaserAlphas_;
  edm::ESGetToken<EcalLinearCorrectionsGPU, EcalLinearCorrectionsRcd> tokenLinearCorrections_;
  edm::ESGetToken<EcalRecHitParametersGPU, JobConfigurationGPURecord> tokenRecHitParameters_;

  // conditions handles
  edm::ESHandle<EcalIntercalibConstantsGPU> IntercalibConstantsHandle_;
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

  // conditions tokens
  tokenADCToGeVConstant_ = esConsumes<EcalRechitADCToGeVConstantGPU, EcalADCToGeVConstantRcd>();
  tokenIntercalibConstants_ = esConsumes<EcalIntercalibConstantsGPU, EcalIntercalibConstantsRcd>();
  tokenChannelStatus_ = esConsumes<EcalRechitChannelStatusGPU, EcalChannelStatusRcd>();
  tokenLaserAPDPNRatios_ = esConsumes<EcalLaserAPDPNRatiosGPU, EcalLaserAPDPNRatiosRcd>();
  tokenLaserAPDPNRatiosRef_ = esConsumes<EcalLaserAPDPNRatiosRefGPU, EcalLaserAPDPNRatiosRefRcd>();
  tokenLaserAlphas_ = esConsumes<EcalLaserAlphasGPU, EcalLaserAlphasRcd>();
  tokenLinearCorrections_ = esConsumes<EcalLinearCorrectionsGPU, EcalLinearCorrectionsRcd>();
  tokenRecHitParameters_ = esConsumes<EcalRecHitParametersGPU, JobConfigurationGPURecord>();
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

  // stop here if there are no uncalibRecHits
  if (neb_ + nee_ == 0)
    return;

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
  IntercalibConstantsHandle_ = setup.getHandle(tokenIntercalibConstants_);
  recHitParametersHandle_ = setup.getHandle(tokenRecHitParameters_);

  auto const& ADCToGeVConstantProduct = setup.getData(tokenADCToGeVConstant_).getProduct(ctx.stream());
  auto const& IntercalibConstantsProduct = IntercalibConstantsHandle_->getProduct(ctx.stream());
  auto const& ChannelStatusProduct = setup.getData(tokenChannelStatus_).getProduct(ctx.stream());

  auto const& LaserAPDPNRatiosProduct = setup.getData(tokenLaserAPDPNRatios_).getProduct(ctx.stream());
  auto const& LaserAPDPNRatiosRefProduct = setup.getData(tokenLaserAPDPNRatiosRef_).getProduct(ctx.stream());
  auto const& LaserAlphasProduct = setup.getData(tokenLaserAlphas_).getProduct(ctx.stream());
  auto const& LinearCorrectionsProduct = setup.getData(tokenLinearCorrections_).getProduct(ctx.stream());
  auto const& recHitParametersProduct = recHitParametersHandle_->getProduct(ctx.stream());

  // set config ptrs : this is done to avoid changing things downstream
  configParameters_.ChannelStatusToBeExcluded = recHitParametersProduct.channelStatusToBeExcluded.get();
  configParameters_.ChannelStatusToBeExcludedSize = std::get<0>(recHitParametersHandle_->getValues()).get().size();
  configParameters_.expanded_v_DB_reco_flags = recHitParametersProduct.expanded_v_DB_reco_flags.get();
  configParameters_.expanded_Sizes_v_DB_reco_flags = recHitParametersProduct.expanded_Sizes_v_DB_reco_flags.get();
  configParameters_.expanded_flagbit_v_DB_reco_flags = recHitParametersProduct.expanded_flagbit_v_DB_reco_flags.get();
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
