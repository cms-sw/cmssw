#include "CUDADataFormats/EcalDigi/interface/DigisCollection.h"
#include "CondFormats/DataRecord/interface/EcalMappingElectronicsRcd.h"
#include "CondFormats/EcalObjects/interface/ElectronicsMappingGPU.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "DeclsForKernels.h"
#include "UnpackGPU.h"

class EcalRawToDigiGPU : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit EcalRawToDigiGPU(edm::ParameterSet const& ps);
  ~EcalRawToDigiGPU() override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, edm::EventSetup const&) override;

private:
  edm::EDGetTokenT<FEDRawDataCollection> rawDataToken_;
  using OutputProduct = cms::cuda::Product<ecal::DigisCollection<calo::common::DevStoragePolicy>>;
  edm::EDPutTokenT<OutputProduct> digisEBToken_, digisEEToken_;
  edm::ESGetToken<ecal::raw::ElectronicsMappingGPU, EcalMappingElectronicsRcd> eMappingToken_;

  cms::cuda::ContextState cudaState_;

  std::vector<int> fedsToUnpack_;

  ecal::raw::ConfigurationParameters config_;
  ecal::raw::OutputDataGPU outputGPU_;
  ecal::raw::OutputDataCPU outputCPU_;
};

void EcalRawToDigiGPU::fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("InputLabel", edm::InputTag("rawDataCollector"));
  std::vector<int> feds(54);
  for (uint32_t i = 0; i < 54; ++i)
    feds[i] = i + 601;
  desc.add<std::vector<int>>("FEDs", feds);
  desc.add<uint32_t>("maxChannelsEB", 61200);
  desc.add<uint32_t>("maxChannelsEE", 14648);
  desc.add<std::string>("digisLabelEB", "ebDigis");
  desc.add<std::string>("digisLabelEE", "eeDigis");

  std::string label = "ecalRawToDigiGPU";
  confDesc.add(label, desc);
}

EcalRawToDigiGPU::EcalRawToDigiGPU(const edm::ParameterSet& ps)
    : rawDataToken_{consumes<FEDRawDataCollection>(ps.getParameter<edm::InputTag>("InputLabel"))},
      digisEBToken_{produces<OutputProduct>(ps.getParameter<std::string>("digisLabelEB"))},
      digisEEToken_{produces<OutputProduct>(ps.getParameter<std::string>("digisLabelEE"))},
      eMappingToken_{esConsumes<ecal::raw::ElectronicsMappingGPU, EcalMappingElectronicsRcd>()},
      fedsToUnpack_{ps.getParameter<std::vector<int>>("FEDs")} {
  config_.maxChannelsEB = ps.getParameter<uint32_t>("maxChannelsEB");
  config_.maxChannelsEE = ps.getParameter<uint32_t>("maxChannelsEE");
}

EcalRawToDigiGPU::~EcalRawToDigiGPU() {}

void EcalRawToDigiGPU::acquire(edm::Event const& event,
                               edm::EventSetup const& setup,
                               edm::WaitingTaskWithArenaHolder holder) {
  // raii
  cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(holder), cudaState_};

  // conditions
  edm::ESHandle<ecal::raw::ElectronicsMappingGPU> eMappingHandle = setup.getHandle(eMappingToken_);
  auto const& eMappingProduct = eMappingHandle->getProduct(ctx.stream());

  // bundle up conditions
  ecal::raw::ConditionsProducts conditions{eMappingProduct};

  // event data
  edm::Handle<FEDRawDataCollection> rawDataHandle;
  event.getByToken(rawDataToken_, rawDataHandle);

  // scratch
  ecal::raw::ScratchDataGPU scratchGPU = {cms::cuda::make_device_unique<uint32_t[]>(2, ctx.stream())};

  // input cpu data
  ecal::raw::InputDataCPU inputCPU = {
      cms::cuda::make_host_unique<unsigned char[]>(ecal::raw::nfeds_max * ecal::raw::nbytes_per_fed_max, ctx.stream()),
      cms::cuda::make_host_unique<uint32_t[]>(ecal::raw::nfeds_max, ctx.stream()),
      cms::cuda::make_host_unique<int[]>(ecal::raw::nfeds_max, ctx.stream())};

  // input data gpu
  ecal::raw::InputDataGPU inputGPU = {cms::cuda::make_device_unique<unsigned char[]>(
                                          ecal::raw::nfeds_max * ecal::raw::nbytes_per_fed_max, ctx.stream()),
                                      cms::cuda::make_device_unique<uint32_t[]>(ecal::raw::nfeds_max, ctx.stream()),
                                      cms::cuda::make_device_unique<int[]>(ecal::raw::nfeds_max, ctx.stream())};

  // output cpu
  outputCPU_ = {cms::cuda::make_host_unique<uint32_t[]>(2, ctx.stream())};

  // output gpu
  outputGPU_.allocate(config_, ctx.stream());

  // iterate over feds
  // TODO: another idea
  //   - loop over all feds to unpack and enqueue cuda memcpy
  //   - accumulate the sizes
  //   - after the loop launch cuda memcpy for sizes
  //   - enqueue the kernel
  uint32_t currentCummOffset = 0;
  uint32_t counter = 0;
  for (auto const& fed : fedsToUnpack_) {
    auto const& data = rawDataHandle->FEDData(fed);
    auto const nbytes = data.size();

    // skip empty feds
    if (nbytes < ecal::raw::empty_event_size)
      continue;

    // copy raw data into plain buffer
    std::memcpy(inputCPU.data.get() + currentCummOffset, data.data(), nbytes);
    // set the offset in bytes from the start
    inputCPU.offsets[counter] = currentCummOffset;
    inputCPU.feds[counter] = fed;

    // this is the current offset into the vector
    currentCummOffset += nbytes;
    ++counter;
  }

  // unpack if at least one FED has data
  if (counter > 0) {
    ecal::raw::entryPoint(
        inputCPU, inputGPU, outputGPU_, scratchGPU, outputCPU_, conditions, ctx.stream(), counter, currentCummOffset);
  }
}

void EcalRawToDigiGPU::produce(edm::Event& event, edm::EventSetup const& setup) {
  cms::cuda::ScopedContextProduce ctx{cudaState_};

  // get the number of channels
  outputGPU_.digisEB.size = outputCPU_.nchannels[0];
  outputGPU_.digisEE.size = outputCPU_.nchannels[1];

  ctx.emplace(event, digisEBToken_, std::move(outputGPU_.digisEB));
  ctx.emplace(event, digisEEToken_, std::move(outputGPU_.digisEE));

  // reset ptrs that are carried as members
  outputCPU_.nchannels.reset();
}

DEFINE_FWK_MODULE(EcalRawToDigiGPU);
