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

class EcalCPUDigisProducer : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit EcalCPUDigisProducer(edm::ParameterSet const& ps);
  ~EcalCPUDigisProducer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, edm::EventSetup const&) override;

private:
  edm::EDGetTokenT<cms::cuda::Product<ecal::DigisCollection>> digisInEBToken_, digisInEEToken_;
  edm::EDPutTokenT<EBDigiCollection> digisOutEBToken_;
  edm::EDPutTokenT<EEDigiCollection> digisOutEEToken_;

  // FIXME better way to pass pointers from acquire to produce?
  std::vector<uint32_t, CUDAHostAllocator<uint32_t>> idsebtmp, idseetmp;
  std::vector<uint16_t, CUDAHostAllocator<uint16_t>> dataebtmp, dataeetmp;
};

void EcalCPUDigisProducer::fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("digisInLabelEB", edm::InputTag{"ecalRawToDigiGPU", "ebDigisGPU"});
  desc.add<edm::InputTag>("digisInLabelEE", edm::InputTag{"ecalRawToDigiGPU", "eeDigisGPU"});
  desc.add<std::string>("digisOutLabelEB", "ebDigis");
  desc.add<std::string>("digisOutLabelEE", "eeDigis");

  std::string label = "ecalCPUDigisProducer";
  confDesc.add(label, desc);
}

EcalCPUDigisProducer::EcalCPUDigisProducer(const edm::ParameterSet& ps)
    : digisInEBToken_{consumes<cms::cuda::Product<ecal::DigisCollection>>(
          ps.getParameter<edm::InputTag>("digisInLabelEB"))},
      digisInEEToken_{
          consumes<cms::cuda::Product<ecal::DigisCollection>>(ps.getParameter<edm::InputTag>("digisInLabelEE"))},
      digisOutEBToken_{produces<EBDigiCollection>(ps.getParameter<std::string>("digisOutLabelEB"))},
      digisOutEEToken_{produces<EEDigiCollection>(ps.getParameter<std::string>("digisOutLabelEE"))} {}

EcalCPUDigisProducer::~EcalCPUDigisProducer() {}

void EcalCPUDigisProducer::acquire(edm::Event const& event,
                                   edm::EventSetup const& setup,
                                   edm::WaitingTaskWithArenaHolder taskHolder) {
  // retrieve data/ctx
  auto const& ebdigisProduct = event.get(digisInEBToken_);
  auto const& eedigisProduct = event.get(digisInEEToken_);
  cms::cuda::ScopedContextAcquire ctx{ebdigisProduct, std::move(taskHolder)};
  auto const& ebdigis = ctx.get(ebdigisProduct);
  auto const& eedigis = ctx.get(eedigisProduct);

  // resize out tmp buffers
  // FIXME remove hardcoded values
  idsebtmp.resize(ebdigis.ndigis);
  dataebtmp.resize(ebdigis.ndigis * 10);
  idseetmp.resize(eedigis.ndigis);
  dataeetmp.resize(eedigis.ndigis * 10);

  // enqeue transfers
  cudaCheck(cudaMemcpyAsync(
      dataebtmp.data(), ebdigis.data, dataebtmp.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost, ctx.stream()));
  cudaCheck(cudaMemcpyAsync(
      dataeetmp.data(), eedigis.data, dataeetmp.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost, ctx.stream()));
  cudaCheck(cudaMemcpyAsync(
      idsebtmp.data(), ebdigis.ids, idsebtmp.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, ctx.stream()));
  cudaCheck(cudaMemcpyAsync(
      idseetmp.data(), eedigis.ids, idseetmp.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, ctx.stream()));
}

void EcalCPUDigisProducer::produce(edm::Event& event, edm::EventSetup const& setup) {
  // output collections
  auto digisEB = std::make_unique<EBDigiCollection>();
  auto digisEE = std::make_unique<EEDigiCollection>();
  digisEB->resize(idsebtmp.size());
  digisEE->resize(idseetmp.size());

  // cast constness away
  // use pointers to buffers instead of move operator= semantics
  // cause we have different allocators in there...
  auto* dataEB = const_cast<uint16_t*>(digisEB->data().data());
  auto* dataEE = const_cast<uint16_t*>(digisEE->data().data());
  auto* idsEB = const_cast<uint32_t*>(digisEB->ids().data());
  auto* idsEE = const_cast<uint32_t*>(digisEE->ids().data());

  // copy data
  std::memcpy(dataEB, dataebtmp.data(), dataebtmp.size() * sizeof(uint16_t));
  std::memcpy(dataEE, dataeetmp.data(), dataeetmp.size() * sizeof(uint16_t));
  std::memcpy(idsEB, idsebtmp.data(), idsebtmp.size() * sizeof(uint32_t));
  std::memcpy(idsEE, idseetmp.data(), idseetmp.size() * sizeof(uint32_t));

  event.put(digisOutEBToken_, std::move(digisEB));
  event.put(digisOutEEToken_, std::move(digisEE));
}

DEFINE_FWK_MODULE(EcalCPUDigisProducer);
