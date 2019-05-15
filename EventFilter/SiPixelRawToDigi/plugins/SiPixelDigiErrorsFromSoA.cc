#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/SiPixelDetId/interface/PixelFEDChannel.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDigi/interface/SiPixelDigiErrorsSoA.h"
#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>

class SiPixelDigiErrorsFromSoA: public edm::stream::EDProducer<> {
public:
  explicit SiPixelDigiErrorsFromSoA(const edm::ParameterSet& iConfig);
  ~SiPixelDigiErrorsFromSoA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<SiPixelDigiErrorsSoA> digiErrorSoAGetToken_;

  edm::EDPutTokenT<edm::DetSetVector<SiPixelRawDataError>> errorPutToken_;
  edm::EDPutTokenT<DetIdCollection> tkErrorPutToken_;
  edm::EDPutTokenT<DetIdCollection> userErrorPutToken_;
  edm::EDPutTokenT<edmNew::DetSetVector<PixelFEDChannel>> disabledChannelPutToken_;

  edm::ESWatcher<SiPixelFedCablingMapRcd> cablingWatcher_;
  std::unique_ptr<SiPixelFedCablingTree> cabling_;
  const std::string cablingMapLabel_;

  const std::vector<int> tkerrorlist_;
  const std::vector<int> usererrorlist_;

  const bool usePhase1_;
};

SiPixelDigiErrorsFromSoA::SiPixelDigiErrorsFromSoA(const edm::ParameterSet& iConfig):
  digiErrorSoAGetToken_{consumes<SiPixelDigiErrorsSoA>(iConfig.getParameter<edm::InputTag>("digiErrorSoASrc"))},
  errorPutToken_{produces<edm::DetSetVector<SiPixelRawDataError>>()},
  tkErrorPutToken_{produces<DetIdCollection>()},
  userErrorPutToken_{produces<DetIdCollection>("UserErrorModules")},
  disabledChannelPutToken_{produces<edmNew::DetSetVector<PixelFEDChannel>>()},
  cablingMapLabel_(iConfig.getParameter<std::string>("CablingMapLabel")),
  tkerrorlist_(iConfig.getParameter<std::vector<int>>("ErrorList")),
  usererrorlist_(iConfig.getParameter<std::vector<int>>("UserErrorList")),
  usePhase1_(iConfig.getParameter<bool> ("UsePhase1"))
{}

void SiPixelDigiErrorsFromSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("digiErrorSoASrc", edm::InputTag("siPixelDigiErrorsSoA"));
  desc.add<std::string>("CablingMapLabel","")->setComment("CablingMap label");
  desc.add<bool>("UsePhase1",false)->setComment("##  Use phase1");
  desc.add<std::vector<int> >("ErrorList", std::vector<int>{29})->setComment("## ErrorList: list of error codes used by tracking to invalidate modules");
  desc.add<std::vector<int> >("UserErrorList", std::vector<int>{40})->setComment("## UserErrorList: list of error codes used by Pixel experts for investigation");
  descriptions.addWithDefaultLabel(desc);
}

void SiPixelDigiErrorsFromSoA::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // pack errors into collection

  // initialize cabling map or update if necessary
  if (cablingWatcher_.check(iSetup)) {
    // cabling map, which maps online address (fed->link->ROC->local pixel) to offline (DetId->global pixel)
    edm::ESTransientHandle<SiPixelFedCablingMap> cablingMap;
    iSetup.get<SiPixelFedCablingMapRcd>().get(cablingMapLabel_, cablingMap);
    cabling_ = cablingMap->cablingTree();
    LogDebug("map version:")<< cabling_->version();
  }

  const auto& digiErrors = iEvent.get(digiErrorSoAGetToken_);


  edm::DetSetVector<SiPixelRawDataError> errorcollection{};
  DetIdCollection tkerror_detidcollection{};
  DetIdCollection usererror_detidcollection{};
  edmNew::DetSetVector<PixelFEDChannel> disabled_channelcollection{};

  PixelDataFormatter formatter(cabling_.get(), usePhase1_); // for phase 1 & 0
  const PixelDataFormatter::Errors *formatterErrors = digiErrors.formatterErrors();
  assert(formatterErrors != nullptr);
  auto errors = *formatterErrors; // make a copy
  PixelDataFormatter::DetErrors nodeterrors;

  auto size = digiErrors.size();
  for (auto i = 0U; i < size; i++) {
    PixelErrorCompact err = digiErrors.error(i);
    if (err.errorType != 0) {
      SiPixelRawDataError error(err.word, err.errorType, err.fedId + 1200);
      errors[err.rawId].push_back(error);
    }
  }

  constexpr uint32_t dummydetid = 0xffffffff;
  typedef PixelDataFormatter::Errors::iterator IE;
  for (IE is = errors.begin(); is != errors.end(); is++) {

    uint32_t errordetid = is->first;
    if (errordetid == dummydetid) {// errors given dummy detId must be sorted by Fed
      nodeterrors.insert( nodeterrors.end(), errors[errordetid].begin(), errors[errordetid].end() );
    }
    else {
      edm::DetSet<SiPixelRawDataError>& errorDetSet = errorcollection.find_or_insert(errordetid);
      errorDetSet.data.insert(errorDetSet.data.end(), is->second.begin(), is->second.end());
      // Fill detid of the detectors where there is error AND the error number is listed
      // in the configurable error list in the job option cfi.
      // Code needs to be here, because there can be a set of errors for each
      // entry in the for loop over PixelDataFormatter::Errors

      std::vector<PixelFEDChannel> disabledChannelsDetSet;

      for (auto const& aPixelError : errorDetSet) {
        // For the time being, we extend the error handling functionality with ErrorType 25
        // In the future, we should sort out how the usage of tkerrorlist can be generalized
        if (aPixelError.getType() == 25) {
          int fedId = aPixelError.getFedId();
          const sipixelobjects::PixelFEDCabling* fed = cabling_->fed(fedId);
          if (fed) {
            cms_uint32_t linkId = formatter.linkId(aPixelError.getWord32());
            const sipixelobjects::PixelFEDLink* link = fed->link(linkId);
            if (link) {
              // The "offline" 0..15 numbering is fixed by definition, also, the FrameConversion depends on it
              // in contrast, the ROC-in-channel numbering is determined by hardware --> better to use the "offline" scheme
              PixelFEDChannel ch = {fed->id(), linkId, 25, 0};
              for (unsigned int iRoc = 1; iRoc <= link->numberOfROCs(); iRoc++) {
                const sipixelobjects::PixelROC * roc = link->roc(iRoc);
                if (roc->idInDetUnit() < ch.roc_first) ch.roc_first = roc->idInDetUnit();
                if (roc->idInDetUnit() > ch.roc_last) ch.roc_last = roc->idInDetUnit();
              }
              if (ch.roc_first<ch.roc_last) disabledChannelsDetSet.push_back(ch);
            }
          }
        }
        else {
          // fill list of detIds to be turned off by tracking
          if (!tkerrorlist_.empty()) {
            auto it_find = std::find(tkerrorlist_.begin(), tkerrorlist_.end(), aPixelError.getType());
            if (it_find != tkerrorlist_.end()) {
              tkerror_detidcollection.push_back(errordetid);
            }
          }
        }
        
        // fill list of detIds with errors to be studied
        if (!usererrorlist_.empty()) {
          auto it_find = std::find(usererrorlist_.begin(), usererrorlist_.end(), aPixelError.getType());
          if (it_find != usererrorlist_.end()) {
            usererror_detidcollection.push_back(errordetid);
          }
        }

      } // loop on DetSet of errors

      if (!disabledChannelsDetSet.empty()) {
        disabled_channelcollection.insert(errordetid, disabledChannelsDetSet.data(), disabledChannelsDetSet.size());
      }

    } // if error assigned to a real DetId
  } // loop on errors in event for this FED

  edm::DetSet<SiPixelRawDataError>& errorDetSet = errorcollection.find_or_insert(dummydetid);
  errorDetSet.data = nodeterrors;

  iEvent.emplace(errorPutToken_, std::move(errorcollection));
  iEvent.emplace(tkErrorPutToken_, std::move(tkerror_detidcollection));
  iEvent.emplace(userErrorPutToken_, std::move(usererror_detidcollection));
  iEvent.emplace(disabledChannelPutToken_, std::move(disabled_channelcollection));
}

DEFINE_FWK_MODULE(SiPixelDigiErrorsFromSoA);
