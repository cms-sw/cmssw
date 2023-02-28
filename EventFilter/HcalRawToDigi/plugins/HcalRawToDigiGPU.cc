#include "CUDADataFormats/Common/interface/Product.h"
#include "CondFormats/DataRecord/interface/HcalElectronicsMapRcd.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"

#include "DeclsForKernels.h"
#include "DecodeGPU.h"
#include "ElectronicsMappingGPU.h"

class HcalRawToDigiGPU : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit HcalRawToDigiGPU(edm::ParameterSet const& ps);
  ~HcalRawToDigiGPU() override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, edm::EventSetup const&) override;

private:
  edm::ESGetToken<hcal::raw::ElectronicsMappingGPU, HcalElectronicsMapRcd> eMappingToken_;
  edm::EDGetTokenT<FEDRawDataCollection> rawDataToken_;
  using ProductTypef01 = cms::cuda::Product<hcal::DigiCollection<hcal::Flavor1, calo::common::DevStoragePolicy>>;
  edm::EDPutTokenT<ProductTypef01> digisF01HEToken_;
  using ProductTypef5 = cms::cuda::Product<hcal::DigiCollection<hcal::Flavor5, calo::common::DevStoragePolicy>>;
  edm::EDPutTokenT<ProductTypef5> digisF5HBToken_;
  using ProductTypef3 = cms::cuda::Product<hcal::DigiCollection<hcal::Flavor3, calo::common::DevStoragePolicy>>;
  edm::EDPutTokenT<ProductTypef3> digisF3HBToken_;

  cms::cuda::ContextState cudaState_;

  std::vector<int> fedsToUnpack_;

  hcal::raw::ConfigurationParameters config_;
  hcal::raw::OutputDataGPU outputGPU_;
  hcal::raw::OutputDataCPU outputCPU_;
};

void HcalRawToDigiGPU::fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("InputLabel", edm::InputTag("rawDataCollector"));
  auto nFeds = FEDNumbering::MAXHCALuTCAFEDID - FEDNumbering::MINHCALuTCAFEDID + 1;
  std::vector<int> feds(nFeds);
  for (int i = 0; i < nFeds; ++i)
    feds[i] = i + FEDNumbering::MINHCALuTCAFEDID;
  desc.add<std::vector<int>>("FEDs", feds);
  desc.add<uint32_t>("maxChannelsF01HE", 10000u);
  desc.add<uint32_t>("maxChannelsF5HB", 10000u);
  desc.add<uint32_t>("maxChannelsF3HB", 10000u);
  desc.add<uint32_t>("nsamplesF01HE", 8);
  desc.add<uint32_t>("nsamplesF5HB", 8);
  desc.add<uint32_t>("nsamplesF3HB", 8);
  desc.add<std::string>("digisLabelF5HB", "f5HBDigisGPU");
  desc.add<std::string>("digisLabelF01HE", "f01HEDigisGPU");
  desc.add<std::string>("digisLabelF3HB", "f3HBDigisGPU");

  std::string label = "hcalRawToDigiGPU";
  confDesc.add(label, desc);
}

HcalRawToDigiGPU::HcalRawToDigiGPU(const edm::ParameterSet& ps)
    : eMappingToken_{esConsumes()},
      rawDataToken_{consumes<FEDRawDataCollection>(ps.getParameter<edm::InputTag>("InputLabel"))},
      digisF01HEToken_{produces<ProductTypef01>(ps.getParameter<std::string>("digisLabelF01HE"))},
      digisF5HBToken_{produces<ProductTypef5>(ps.getParameter<std::string>("digisLabelF5HB"))},
      digisF3HBToken_{produces<ProductTypef3>(ps.getParameter<std::string>("digisLabelF3HB"))},
      fedsToUnpack_{ps.getParameter<std::vector<int>>("FEDs")} {
  config_.maxChannelsF01HE = ps.getParameter<uint32_t>("maxChannelsF01HE");
  config_.maxChannelsF5HB = ps.getParameter<uint32_t>("maxChannelsF5HB");
  config_.maxChannelsF3HB = ps.getParameter<uint32_t>("maxChannelsF3HB");
  config_.nsamplesF01HE = ps.getParameter<uint32_t>("nsamplesF01HE");
  config_.nsamplesF5HB = ps.getParameter<uint32_t>("nsamplesF5HB");
  config_.nsamplesF3HB = ps.getParameter<uint32_t>("nsamplesF3HB");
}

HcalRawToDigiGPU::~HcalRawToDigiGPU() {}

void HcalRawToDigiGPU::acquire(edm::Event const& event,
                               edm::EventSetup const& setup,
                               edm::WaitingTaskWithArenaHolder holder) {
  // raii
  cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(holder), cudaState_};

  // conditions
  auto const& eMappingProduct = setup.getData(eMappingToken_).getProduct(ctx.stream());

  // bundle up conditions
  hcal::raw::ConditionsProducts conditions{eMappingProduct};

  // event data
  edm::Handle<FEDRawDataCollection> rawDataHandle;
  event.getByToken(rawDataToken_, rawDataHandle);

  // scratch
  hcal::raw::ScratchDataGPU scratchGPU = {
      cms::cuda::make_device_unique<uint32_t[]>(hcal::raw::numOutputCollections, ctx.stream())};

  // input cpu data
  hcal::raw::InputDataCPU inputCPU = {cms::cuda::make_host_unique<unsigned char[]>(
                                          hcal::raw::utca_nfeds_max * hcal::raw::nbytes_per_fed_max, ctx.stream()),
                                      cms::cuda::make_host_unique<uint32_t[]>(hcal::raw::utca_nfeds_max, ctx.stream()),
                                      cms::cuda::make_host_unique<int[]>(hcal::raw::utca_nfeds_max, ctx.stream())};

  // input data gpu
  hcal::raw::InputDataGPU inputGPU = {
      cms::cuda::make_device_unique<unsigned char[]>(hcal::raw::utca_nfeds_max * hcal::raw::nbytes_per_fed_max,
                                                     ctx.stream()),
      cms::cuda::make_device_unique<uint32_t[]>(hcal::raw::utca_nfeds_max, ctx.stream()),
      cms::cuda::make_device_unique<int[]>(hcal::raw::utca_nfeds_max, ctx.stream())};

  // output cpu
  outputCPU_ = {cms::cuda::make_host_unique<uint32_t[]>(hcal::raw::numOutputCollections, ctx.stream())};

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
    if (nbytes < hcal::raw::empty_event_size)
      continue;

#ifdef HCAL_RAWDECODE_CPUDEBUG
    printf("fed = %d nbytes = %lu\n", fed, nbytes);
#endif

    // copy raw data into plain buffer
    std::memcpy(inputCPU.data.get() + currentCummOffset, data.data(), nbytes);
    // set the offset in bytes from the start
    inputCPU.offsets[counter] = currentCummOffset;
    inputCPU.feds[counter] = fed;

    // this is the current offset into the vector
    currentCummOffset += nbytes;
    ++counter;
  }

  hcal::raw::entryPoint(inputCPU,
                        inputGPU,
                        outputGPU_,
                        scratchGPU,
                        outputCPU_,
                        conditions,
                        config_,
                        ctx.stream(),
                        counter,
                        currentCummOffset);
}

void HcalRawToDigiGPU::produce(edm::Event& event, edm::EventSetup const& setup) {
  cms::cuda::ScopedContextProduce ctx{cudaState_};

#ifdef HCAL_RAWDECODE_CPUDEBUG
  printf("f01he channels = %u f5hb channesl = %u\n",
         outputCPU_.nchannels[hcal::raw::OutputF01HE],
         outputCPU_.nchannels[hcal::raw::OutputF5HB]);
#endif

  // FIXME: use sizes of views directly for cuda mem cpy?
  auto const nchannelsF01HE = outputCPU_.nchannels[hcal::raw::OutputF01HE];
  auto const nchannelsF5HB = outputCPU_.nchannels[hcal::raw::OutputF5HB];
  auto const nchannelsF3HB = outputCPU_.nchannels[hcal::raw::OutputF3HB];
  outputGPU_.digisF01HE.size = nchannelsF01HE;
  outputGPU_.digisF5HB.size = nchannelsF5HB;
  outputGPU_.digisF3HB.size = nchannelsF3HB;
  outputGPU_.digisF01HE.stride = hcal::compute_stride<hcal::Flavor1>(config_.nsamplesF01HE);
  outputGPU_.digisF5HB.stride = hcal::compute_stride<hcal::Flavor5>(config_.nsamplesF5HB);
  outputGPU_.digisF3HB.stride = hcal::compute_stride<hcal::Flavor3>(config_.nsamplesF3HB);

  ctx.emplace(event, digisF01HEToken_, std::move(outputGPU_.digisF01HE));
  ctx.emplace(event, digisF5HBToken_, std::move(outputGPU_.digisF5HB));
  ctx.emplace(event, digisF3HBToken_, std::move(outputGPU_.digisF3HB));

  // reset ptrs that are carried as members
  outputCPU_.nchannels.reset();
}

DEFINE_FWK_MODULE(HcalRawToDigiGPU);
