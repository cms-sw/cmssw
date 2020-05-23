#include <iostream>

#include "CUDADataFormats/EcalDigi/interface/DigisCollection.h"
#include "CondFormats/DataRecord/interface/EcalMappingElectronicsRcd.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "EventFilter/EcalRawToDigi/interface/DeclsForKernels.h"
#include "EventFilter/EcalRawToDigi/interface/ElectronicsMappingGPU.h"
#include "EventFilter/EcalRawToDigi/interface/UnpackGPU.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

class EcalCPUDigisProducer : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit EcalCPUDigisProducer(edm::ParameterSet const& ps);
  ~EcalCPUDigisProducer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, edm::EventSetup const&) override;

private:
  // input digi collections in GPU-friendly format
  edm::EDGetTokenT<cms::cuda::Product<ecal::DigisCollection>> digisInEBToken_;
  edm::EDGetTokenT<cms::cuda::Product<ecal::DigisCollection>> digisInEEToken_;

  // output digi collections in legacy format
  edm::EDPutTokenT<EBDigiCollection> digisOutEBToken_;
  edm::EDPutTokenT<EEDigiCollection> digisOutEEToken_;

  // whether to produce dummy integrity collections
  bool produceDummyIntegrityCollections_;

  // dummy SRP collections
  edm::EDPutTokenT<EBSrFlagCollection> ebSrFlagToken_;
  edm::EDPutTokenT<EESrFlagCollection> eeSrFlagToken_;

  // dummy integrity for xtal data
  edm::EDPutTokenT<EBDetIdCollection> ebIntegrityGainErrorsToken_;
  edm::EDPutTokenT<EBDetIdCollection> ebIntegrityGainSwitchErrorsToken_;
  edm::EDPutTokenT<EBDetIdCollection> ebIntegrityChIdErrorsToken_;

  // dummy integrity for xtal data - EE specific (to be rivisited towards EB+EE common collection)
  edm::EDPutTokenT<EEDetIdCollection> eeIntegrityGainErrorsToken_;
  edm::EDPutTokenT<EEDetIdCollection> eeIntegrityGainSwitchErrorsToken_;
  edm::EDPutTokenT<EEDetIdCollection> eeIntegrityChIdErrorsToken_;

  // dummy integrity errors
  edm::EDPutTokenT<EcalElectronicsIdCollection> integrityTTIdErrorsToken_;
  edm::EDPutTokenT<EcalElectronicsIdCollection> integrityBlockSizeErrorsToken_;

  // FIXME better way to pass pointers from acquire to produce?
  std::vector<uint32_t, cms::cuda::HostAllocator<uint32_t>> idsebtmp, idseetmp;
  std::vector<uint16_t, cms::cuda::HostAllocator<uint16_t>> dataebtmp, dataeetmp;
};

void EcalCPUDigisProducer::fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("digisInLabelEB", edm::InputTag{"ecalRawToDigiGPU", "ebDigisGPU"});
  desc.add<edm::InputTag>("digisInLabelEE", edm::InputTag{"ecalRawToDigiGPU", "eeDigisGPU"});
  desc.add<std::string>("digisOutLabelEB", "ebDigis");
  desc.add<std::string>("digisOutLabelEE", "eeDigis");

  desc.add<bool>("produceDummyIntegrityCollections", false);

  std::string label = "ecalCPUDigisProducer";
  confDesc.add(label, desc);
}

EcalCPUDigisProducer::EcalCPUDigisProducer(const edm::ParameterSet& ps)
    :  // input digi collections in GPU-friendly format
      digisInEBToken_{
          consumes<cms::cuda::Product<ecal::DigisCollection>>(ps.getParameter<edm::InputTag>("digisInLabelEB"))},
      digisInEEToken_{
          consumes<cms::cuda::Product<ecal::DigisCollection>>(ps.getParameter<edm::InputTag>("digisInLabelEE"))},
      // output digi collections in legacy format
      digisOutEBToken_{produces<EBDigiCollection>(ps.getParameter<std::string>("digisOutLabelEB"))},
      digisOutEEToken_{produces<EEDigiCollection>(ps.getParameter<std::string>("digisOutLabelEE"))},
      // whether to produce dummy integrity collections
      produceDummyIntegrityCollections_{ps.getParameter<bool>("produceDummyIntegrityCollections")},
      // dummy SRP collections
      ebSrFlagToken_{produceDummyIntegrityCollections_ ? produces<EBSrFlagCollection>()
                                                       : edm::EDPutTokenT<EBSrFlagCollection>{}},
      eeSrFlagToken_{produceDummyIntegrityCollections_ ? produces<EESrFlagCollection>()
                                                       : edm::EDPutTokenT<EESrFlagCollection>{}},
      // dummy integrity for xtal data
      ebIntegrityGainErrorsToken_{produceDummyIntegrityCollections_
                                      ? produces<EBDetIdCollection>("EcalIntegrityGainErrors")
                                      : edm::EDPutTokenT<EBDetIdCollection>{}},
      ebIntegrityGainSwitchErrorsToken_{produceDummyIntegrityCollections_
                                            ? produces<EBDetIdCollection>("EcalIntegrityGainSwitchErrors")
                                            : edm::EDPutTokenT<EBDetIdCollection>{}},
      ebIntegrityChIdErrorsToken_{produceDummyIntegrityCollections_
                                      ? produces<EBDetIdCollection>("EcalIntegrityChIdErrors")
                                      : edm::EDPutTokenT<EBDetIdCollection>{}},
      // dummy integrity for xtal data - EE specific (to be rivisited towards EB+EE common collection)
      eeIntegrityGainErrorsToken_{produceDummyIntegrityCollections_
                                      ? produces<EEDetIdCollection>("EcalIntegrityGainErrors")
                                      : edm::EDPutTokenT<EEDetIdCollection>{}},
      eeIntegrityGainSwitchErrorsToken_{produceDummyIntegrityCollections_
                                            ? produces<EEDetIdCollection>("EcalIntegrityGainSwitchErrors")
                                            : edm::EDPutTokenT<EEDetIdCollection>{}},
      eeIntegrityChIdErrorsToken_{produceDummyIntegrityCollections_
                                      ? produces<EEDetIdCollection>("EcalIntegrityChIdErrors")
                                      : edm::EDPutTokenT<EEDetIdCollection>{}},
      // dummy integrity errors
      integrityTTIdErrorsToken_{produceDummyIntegrityCollections_
                                    ? produces<EcalElectronicsIdCollection>("EcalIntegrityTTIdErrors")
                                    : edm::EDPutTokenT<EcalElectronicsIdCollection>{}},
      integrityBlockSizeErrorsToken_{produceDummyIntegrityCollections_
                                         ? produces<EcalElectronicsIdCollection>("EcalIntegrityBlockSizeErrors")
                                         : edm::EDPutTokenT<EcalElectronicsIdCollection>{}} {}

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

  if (produceDummyIntegrityCollections_) {
    // dummy SRP collections
    event.emplace(ebSrFlagToken_);
    event.emplace(eeSrFlagToken_);
    // dummy integrity for xtal data
    event.emplace(ebIntegrityGainErrorsToken_);
    event.emplace(ebIntegrityGainSwitchErrorsToken_);
    event.emplace(ebIntegrityChIdErrorsToken_);
    // dummy integrity for xtal data - EE specific (to be rivisited towards EB+EE common collection)
    event.emplace(eeIntegrityGainErrorsToken_);
    event.emplace(eeIntegrityGainSwitchErrorsToken_);
    event.emplace(eeIntegrityChIdErrorsToken_);
    // dummy integrity errors
    event.emplace(integrityTTIdErrorsToken_);
    event.emplace(integrityBlockSizeErrorsToken_);
  }
}

DEFINE_FWK_MODULE(EcalCPUDigisProducer);
