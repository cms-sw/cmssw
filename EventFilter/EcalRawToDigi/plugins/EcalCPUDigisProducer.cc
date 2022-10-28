#include <iostream>
#include <utility>

#include "CUDADataFormats/EcalDigi/interface/DigisCollection.h"
#include "CondFormats/DataRecord/interface/EcalMappingElectronicsRcd.h"
#include "CondFormats/EcalObjects/interface/ElectronicsMappingGPU.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "DeclsForKernels.h"
#include "UnpackGPU.h"

class EcalCPUDigisProducer : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit EcalCPUDigisProducer(edm::ParameterSet const& ps);
  ~EcalCPUDigisProducer() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, edm::EventSetup const&) override;

  template <typename ProductType, typename... ARGS>
  edm::EDPutTokenT<ProductType> dummyProduces(ARGS&&... args) {
    return (produceDummyIntegrityCollections_) ? produces<ProductType>(std::forward<ARGS>(args)...)
                                               : edm::EDPutTokenT<ProductType>{};
  }

private:
  // input digi collections in GPU-friendly format
  using InputProduct = cms::cuda::Product<ecal::DigisCollection<calo::common::DevStoragePolicy>>;
  edm::EDGetTokenT<InputProduct> digisInEBToken_;
  edm::EDGetTokenT<InputProduct> digisInEEToken_;

  // output digi collections in legacy format
  edm::EDPutTokenT<EBDigiCollection> digisOutEBToken_;
  edm::EDPutTokenT<EEDigiCollection> digisOutEEToken_;

  // whether to produce dummy integrity collections
  bool produceDummyIntegrityCollections_;

  // dummy producer collections
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
  edm::EDPutTokenT<EcalElectronicsIdCollection> integrityZSXtalIdErrorsToken_;
  edm::EDPutTokenT<EcalElectronicsIdCollection> integrityBlockSizeErrorsToken_;

  edm::EDPutTokenT<EcalPnDiodeDigiCollection> pnDiodeDigisToken_;

  // dummy TCC collections
  edm::EDPutTokenT<EcalTrigPrimDigiCollection> ecalTriggerPrimitivesToken_;
  edm::EDPutTokenT<EcalPSInputDigiCollection> ecalPseudoStripInputsToken_;

  // FIXME better way to pass pointers from acquire to produce?
  std::vector<uint32_t, cms::cuda::HostAllocator<uint32_t>> idsebtmp, idseetmp;
  std::vector<uint16_t, cms::cuda::HostAllocator<uint16_t>> dataebtmp, dataeetmp;
};

void EcalCPUDigisProducer::fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("digisInLabelEB", edm::InputTag{"ecalRawToDigiGPU", "ebDigis"});
  desc.add<edm::InputTag>("digisInLabelEE", edm::InputTag{"ecalRawToDigiGPU", "eeDigis"});
  desc.add<std::string>("digisOutLabelEB", "ebDigis");
  desc.add<std::string>("digisOutLabelEE", "eeDigis");

  desc.add<bool>("produceDummyIntegrityCollections", false);

  std::string label = "ecalCPUDigisProducer";
  confDesc.add(label, desc);
}

EcalCPUDigisProducer::EcalCPUDigisProducer(const edm::ParameterSet& ps)
    :  // input digi collections in GPU-friendly format
      digisInEBToken_{consumes<InputProduct>(ps.getParameter<edm::InputTag>("digisInLabelEB"))},
      digisInEEToken_{consumes<InputProduct>(ps.getParameter<edm::InputTag>("digisInLabelEE"))},

      // output digi collections in legacy format
      digisOutEBToken_{produces<EBDigiCollection>(ps.getParameter<std::string>("digisOutLabelEB"))},
      digisOutEEToken_{produces<EEDigiCollection>(ps.getParameter<std::string>("digisOutLabelEE"))},

      // whether to produce dummy integrity collections
      produceDummyIntegrityCollections_{ps.getParameter<bool>("produceDummyIntegrityCollections")},

      // dummy collections
      ebSrFlagToken_{dummyProduces<EBSrFlagCollection>()},
      eeSrFlagToken_{dummyProduces<EESrFlagCollection>()},

      // dummy integrity for xtal data
      ebIntegrityGainErrorsToken_{dummyProduces<EBDetIdCollection>("EcalIntegrityGainErrors")},
      ebIntegrityGainSwitchErrorsToken_{dummyProduces<EBDetIdCollection>("EcalIntegrityGainSwitchErrors")},
      ebIntegrityChIdErrorsToken_{dummyProduces<EBDetIdCollection>("EcalIntegrityChIdErrors")},

      // dummy integrity for xtal data - EE specific (to be rivisited towards EB+EE common collection)
      eeIntegrityGainErrorsToken_{dummyProduces<EEDetIdCollection>("EcalIntegrityGainErrors")},
      eeIntegrityGainSwitchErrorsToken_{dummyProduces<EEDetIdCollection>("EcalIntegrityGainSwitchErrors")},
      eeIntegrityChIdErrorsToken_{dummyProduces<EEDetIdCollection>("EcalIntegrityChIdErrors")},

      // dummy integrity errors
      integrityTTIdErrorsToken_{dummyProduces<EcalElectronicsIdCollection>("EcalIntegrityTTIdErrors")},
      integrityZSXtalIdErrorsToken_{dummyProduces<EcalElectronicsIdCollection>("EcalIntegrityZSXtalIdErrors")},
      integrityBlockSizeErrorsToken_{dummyProduces<EcalElectronicsIdCollection>("EcalIntegrityBlockSizeErrors")},

      //
      pnDiodeDigisToken_{dummyProduces<EcalPnDiodeDigiCollection>()},

      // dummy TCC collections
      ecalTriggerPrimitivesToken_{dummyProduces<EcalTrigPrimDigiCollection>("EcalTriggerPrimitives")},
      ecalPseudoStripInputsToken_{dummyProduces<EcalPSInputDigiCollection>("EcalPseudoStripInputs")}
// constructor body
{}

void EcalCPUDigisProducer::acquire(edm::Event const& event,
                                   edm::EventSetup const& setup,
                                   edm::WaitingTaskWithArenaHolder taskHolder) {
  // retrieve data/ctx
  auto const& ebdigisProduct = event.get(digisInEBToken_);
  auto const& eedigisProduct = event.get(digisInEEToken_);
  cms::cuda::ScopedContextAcquire ctx{ebdigisProduct, std::move(taskHolder)};
  auto const& ebdigis = ctx.get(ebdigisProduct);
  auto const& eedigis = ctx.get(eedigisProduct);

  // resize tmp buffers
  dataebtmp.resize(ebdigis.size * EcalDataFrame::MAXSAMPLES);
  dataeetmp.resize(eedigis.size * EcalDataFrame::MAXSAMPLES);
  idsebtmp.resize(ebdigis.size);
  idseetmp.resize(eedigis.size);

  // enqeue transfers
  cudaCheck(cudaMemcpyAsync(
      dataebtmp.data(), ebdigis.data.get(), dataebtmp.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost, ctx.stream()));
  cudaCheck(cudaMemcpyAsync(
      dataeetmp.data(), eedigis.data.get(), dataeetmp.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost, ctx.stream()));
  cudaCheck(cudaMemcpyAsync(
      idsebtmp.data(), ebdigis.ids.get(), idsebtmp.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, ctx.stream()));
  cudaCheck(cudaMemcpyAsync(
      idseetmp.data(), eedigis.ids.get(), idseetmp.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, ctx.stream()));
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

  digisEB->sort();
  digisEE->sort();

  event.put(digisOutEBToken_, std::move(digisEB));
  event.put(digisOutEEToken_, std::move(digisEE));

  if (produceDummyIntegrityCollections_) {
    // dummy collections
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
    event.emplace(integrityZSXtalIdErrorsToken_);
    event.emplace(integrityBlockSizeErrorsToken_);
    //
    event.emplace(pnDiodeDigisToken_);
    // dummy TCC collections
    event.emplace(ecalTriggerPrimitivesToken_);
    event.emplace(ecalPseudoStripInputsToken_);
  }
}

DEFINE_FWK_MODULE(EcalCPUDigisProducer);
