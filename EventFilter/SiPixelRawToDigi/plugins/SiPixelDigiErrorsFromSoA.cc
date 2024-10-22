#include <memory>

#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/SiPixelDetId/interface/PixelFEDChannel.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelErrorsSoA.h"
#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class SiPixelDigiErrorsFromSoA : public edm::stream::EDProducer<> {
public:
  explicit SiPixelDigiErrorsFromSoA(const edm::ParameterSet& iConfig);
  ~SiPixelDigiErrorsFromSoA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  const edm::ESGetToken<SiPixelFedCablingMap, SiPixelFedCablingMapRcd> cablingToken_;
  const edm::EDGetTokenT<SiPixelErrorsSoA> digiErrorSoAGetToken_;
  const edm::EDPutTokenT<edm::DetSetVector<SiPixelRawDataError>> errorPutToken_;
  const edm::EDPutTokenT<DetIdCollection> tkErrorPutToken_;
  const edm::EDPutTokenT<DetIdCollection> userErrorPutToken_;
  const edm::EDPutTokenT<edmNew::DetSetVector<PixelFEDChannel>> disabledChannelPutToken_;

  edm::ESWatcher<SiPixelFedCablingMapRcd> cablingWatcher_;
  std::unique_ptr<SiPixelFedCablingTree> cabling_;

  const std::vector<int> tkerrorlist_;
  const std::vector<int> usererrorlist_;

  const bool usePhase1_;
};

SiPixelDigiErrorsFromSoA::SiPixelDigiErrorsFromSoA(const edm::ParameterSet& iConfig)
    : cablingToken_(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("CablingMapLabel")))),
      digiErrorSoAGetToken_{consumes<SiPixelErrorsSoA>(iConfig.getParameter<edm::InputTag>("digiErrorSoASrc"))},
      errorPutToken_{produces<edm::DetSetVector<SiPixelRawDataError>>()},
      tkErrorPutToken_{produces<DetIdCollection>()},
      userErrorPutToken_{produces<DetIdCollection>("UserErrorModules")},
      disabledChannelPutToken_{produces<edmNew::DetSetVector<PixelFEDChannel>>()},
      tkerrorlist_(iConfig.getParameter<std::vector<int>>("ErrorList")),
      usererrorlist_(iConfig.getParameter<std::vector<int>>("UserErrorList")),
      usePhase1_(iConfig.getParameter<bool>("UsePhase1")) {}

void SiPixelDigiErrorsFromSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("digiErrorSoASrc", edm::InputTag("siPixelDigiErrorsSoA"));
  // the configuration parameters here are named following those in SiPixelRawToDigi
  desc.add<std::string>("CablingMapLabel", "")->setComment("CablingMap label");
  desc.add<bool>("UsePhase1", false)->setComment("##  Use phase1");
  desc.add<std::vector<int>>("ErrorList", std::vector<int>{29})
      ->setComment("## ErrorList: list of error codes used by tracking to invalidate modules");
  desc.add<std::vector<int>>("UserErrorList", std::vector<int>{40})
      ->setComment("## UserErrorList: list of error codes used by Pixel experts for investigation");
  descriptions.addWithDefaultLabel(desc);
}

void SiPixelDigiErrorsFromSoA::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // pack errors into collection

  // initialize cabling map or update if necessary
  if (cablingWatcher_.check(iSetup)) {
    // cabling map, which maps online address (fed->link->ROC->local pixel) to offline (DetId->global pixel)
    const SiPixelFedCablingMap* cablingMap = &iSetup.getData(cablingToken_);
    cabling_ = cablingMap->cablingTree();
    LogDebug("map version:") << cabling_->version();
  }

  const auto& digiErrors = iEvent.get(digiErrorSoAGetToken_);

  edm::DetSetVector<SiPixelRawDataError> errorcollection{};
  DetIdCollection tkerror_detidcollection{};
  DetIdCollection usererror_detidcollection{};
  edmNew::DetSetVector<PixelFEDChannel> disabled_channelcollection{};

  PixelDataFormatter formatter(cabling_.get(), usePhase1_);  // for phase 1 & 0
  const PixelDataFormatter::Errors* formatterErrors = digiErrors.formatterErrors();
  assert(formatterErrors != nullptr);
  auto errors = *formatterErrors;  // make a copy
  PixelDataFormatter::DetErrors nodeterrors;

  auto size = digiErrors.size();
  for (auto i = 0U; i < size; i++) {
    SiPixelErrorCompact err = digiErrors.error(i);
    if (err.errorType != 0) {
      SiPixelRawDataError error(err.word, err.errorType, err.fedId + FEDNumbering::MINSiPixeluTCAFEDID);
      errors[err.rawId].push_back(error);
    }
  }

  formatter.unpackFEDErrors(errors,
                            tkerrorlist_,
                            usererrorlist_,
                            errorcollection,
                            tkerror_detidcollection,
                            usererror_detidcollection,
                            disabled_channelcollection,
                            nodeterrors);

  const uint32_t dummydetid = 0xffffffff;
  edm::DetSet<SiPixelRawDataError>& errorDetSet = errorcollection.find_or_insert(dummydetid);
  errorDetSet.data = nodeterrors;

  iEvent.emplace(errorPutToken_, std::move(errorcollection));
  iEvent.emplace(tkErrorPutToken_, std::move(tkerror_detidcollection));
  iEvent.emplace(userErrorPutToken_, std::move(usererror_detidcollection));
  iEvent.emplace(disabledChannelPutToken_, std::move(disabled_channelcollection));
}

DEFINE_FWK_MODULE(SiPixelDigiErrorsFromSoA);
