#include <iostream>

// framework
#include "FWCore/Framework/interface/stream/EDProducer.h"
//#include "HeterogeneousCore/Producer/interface/HeterogeneousEDProducer.h"
//#include "HeterogeneousCore/Producer/interface/HeterogeneousEvent.h"

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

// algorithm specific

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "CUDADataFormats/EcalDigi/interface/DigisCollection.h"

#include "CondFormats/DataRecord/interface/EcalMappingElectronicsRcd.h"

#include "EventFilter/EcalRawToDigi/interface/ElectronicsMappingGPU.h"

#include "EventFilter/EcalRawToDigi/interface/DeclsForKernels.h"
#include "EventFilter/EcalRawToDigi/interface/UnpackGPU.h"

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
  edm::EDPutTokenT<cms::cuda::Product<ecal::DigisCollection>> digisEBToken_, digisEEToken_;

  cms::cuda::ContextState cudaState_;

  std::vector<int> fedsToUnpack_;

  ecal::raw::ConfigurationParameters config_;
  // FIXME move this to use raii
  ecal::raw::InputDataCPU inputCPU_;
  ecal::raw::InputDataGPU inputGPU_;
  ecal::raw::OutputDataGPU outputGPU_;
  ecal::raw::ScratchDataGPU scratchGPU_;
  ecal::raw::OutputDataCPU outputCPU_;
};

void EcalRawToDigiGPU::fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("InputLabel", edm::InputTag("rawDataCollector"));
  std::vector<int> feds(54);
  for (uint32_t i = 0; i < 54; ++i)
    feds[i] = i + 601;
  desc.add<std::vector<int>>("FEDs", feds);
  desc.add<uint32_t>("maxChannels", 20000);
  desc.add<std::string>("digisLabelEB", "ebDigisGPU");
  desc.add<std::string>("digisLabelEE", "eeDigisGPU");

  std::string label = "ecalRawToDigiGPU";
  confDesc.add(label, desc);
}

EcalRawToDigiGPU::EcalRawToDigiGPU(const edm::ParameterSet& ps)
    : rawDataToken_{consumes<FEDRawDataCollection>(ps.getParameter<edm::InputTag>("InputLabel"))},
      digisEBToken_{produces<cms::cuda::Product<ecal::DigisCollection>>(ps.getParameter<std::string>("digisLabelEB"))},
      digisEEToken_{produces<cms::cuda::Product<ecal::DigisCollection>>(ps.getParameter<std::string>("digisLabelEE"))},
      fedsToUnpack_{ps.getParameter<std::vector<int>>("FEDs")} {
  config_.maxChannels = ps.getParameter<uint32_t>("maxChannels");

  inputCPU_.allocate();
  inputGPU_.allocate();
  outputGPU_.allocate(config_);
  scratchGPU_.allocate(config_);
  outputCPU_.allocate();
}

EcalRawToDigiGPU::~EcalRawToDigiGPU() {
  inputGPU_.deallocate();
  outputGPU_.deallocate(config_);
  scratchGPU_.deallocate(config_);
}

void EcalRawToDigiGPU::acquire(edm::Event const& event,
                               edm::EventSetup const& setup,
                               edm::WaitingTaskWithArenaHolder holder) {
  // raii
  cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(holder), cudaState_};

  // conditions
  edm::ESHandle<ecal::raw::ElectronicsMappingGPU> eMappingHandle;
  setup.get<EcalMappingElectronicsRcd>().get(eMappingHandle);
  auto const& eMappingProduct = eMappingHandle->getProduct(ctx.stream());

  // bundle up conditions
  ecal::raw::ConditionsProducts conditions{eMappingProduct};

  // event data
  edm::Handle<FEDRawDataCollection> rawDataHandle;
  event.getByToken(rawDataToken_, rawDataHandle);

  // iterate over feds
  // TODO: another idea
  //   - loop over all feds to unpack and enqueue cuda memcpy
  //   - accumulate the sizes
  //   - after the loop launch cuda memcpy for sizes
  //   - enqueue the kernel
  uint32_t currentCummOffset = 0;
  uint32_t counter = 0;
  for (auto const& fed : fedsToUnpack_) {
    //std::cout << "fed: " << fed << std::endl;
    auto const& data = rawDataHandle->FEDData(fed);
    auto const nbytes = data.size();

    // skip empty feds
    if (nbytes < ecal::raw::empty_event_size)
      continue;

    // copy raw data into plain buffer
    std::memcpy(inputCPU_.data.data() + currentCummOffset, data.data(), nbytes);
    // set the offset in bytes from the start
    inputCPU_.offsets[counter] = currentCummOffset;
    inputCPU_.feds[counter] = fed;

    // this is the current offset into the vector
    currentCummOffset += nbytes;
    ++counter;
  }

  ecal::raw::entryPoint(
      inputCPU_, inputGPU_, outputGPU_, scratchGPU_, outputCPU_, conditions, ctx.stream(), counter, currentCummOffset);
}

void EcalRawToDigiGPU::produce(edm::Event& event, edm::EventSetup const& setup) {
  cms::cuda::ScopedContextProduce ctx{cudaState_};

  // get the number of channels
  auto const nchannelsEB = outputCPU_.nchannels[0];
  auto const nchannelsEE = outputCPU_.nchannels[1];

  ecal::DigisCollection digisEB{outputGPU_.idsEB, outputGPU_.samplesEB, nchannelsEB};
  ecal::DigisCollection digisEE{outputGPU_.idsEE, outputGPU_.samplesEE, nchannelsEE};

  ctx.emplace(event, digisEBToken_, std::move(digisEB));
  ctx.emplace(event, digisEEToken_, std::move(digisEE));
}

DEFINE_FWK_MODULE(EcalRawToDigiGPU);
