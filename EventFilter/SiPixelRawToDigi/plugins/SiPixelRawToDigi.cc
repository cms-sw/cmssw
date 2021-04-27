// Skip FED40 pilot-blade
// Include parameter driven interface to SiPixelQuality for study purposes
// exclude ROC(raw) based on bad ROC list in SiPixelQuality
// enabled by: process.siPixelDigis.useQuality_Info = True (BY DEFAULT NOT USED)
// 20-10-2010 Andrew York (Tennessee)
// Jan 2016 Tamas Almos Vami (Tav) (Wigner RCP) -- Cabling Map label option
// Jul 2017 Viktor Veszpremi -- added PixelFEDChannel

#include "SiPixelRawToDigi.h"

#include <memory>

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"

#include "EventFilter/SiPixelRawToDigi/interface/PixelUnpackingRegions.h"

using namespace std;

// -----------------------------------------------------------------------------
SiPixelRawToDigi::SiPixelRawToDigi(const edm::ParameterSet& conf)
    : config_(conf),
      badPixelInfo_(nullptr),
      regions_(nullptr),
      tkerrorlist_(config_.getParameter<std::vector<int>>("ErrorList")),
      usererrorlist_(config_.getParameter<std::vector<int>>("UserErrorList")),
      fedRawDataCollectionToken_{consumes<FEDRawDataCollection>(config_.getParameter<edm::InputTag>("InputLabel"))},
      cablingMapToken_{esConsumes<SiPixelFedCablingMap, SiPixelFedCablingMapRcd>(
          edm::ESInputTag("", config_.getParameter<std::string>("CablingMapLabel")))},
      siPixelDigiCollectionToken_{produces<edm::DetSetVector<PixelDigi>>()},
      includeErrors_(config_.getParameter<bool>("IncludeErrors")),
      useQuality_(config_.getParameter<bool>("UseQualityInfo")),
      usePilotBlade_(config_.getParameter<bool>("UsePilotBlade")),
      usePhase1_(config_.getParameter<bool>("UsePhase1"))

{
  if (useQuality_) {
    siPixelQualityToken_ = esConsumes<SiPixelQuality, SiPixelQualityRcd>();
  }

  // Products
  if (includeErrors_) {
    errorPutToken_ = produces<edm::DetSetVector<SiPixelRawDataError>>();
    tkErrorPutToken_ = produces<DetIdCollection>();
    userErrorPutToken_ = produces<DetIdCollection>("UserErrorModules");
    disabledChannelPutToken_ = produces<edmNew::DetSetVector<PixelFEDChannel>>();
  }

  // regions
  if (!config_.getParameter<edm::ParameterSet>("Regions").getParameterNames().empty()) {
    regions_ = new PixelUnpackingRegions(config_, consumesCollector());
  }

  // Control the usage of pilot-blade data, FED=40
  if (usePilotBlade_)
    edm::LogInfo("SiPixelRawToDigi") << " Use pilot blade data (FED 40)";

  // Control the usage of phase1
  if (usePhase1_)
    edm::LogInfo("SiPixelRawToDigi") << " Using phase1";
}

// -----------------------------------------------------------------------------
SiPixelRawToDigi::~SiPixelRawToDigi() {
  edm::LogInfo("SiPixelRawToDigi") << " HERE ** SiPixelRawToDigi destructor!";
  if (regions_)
    delete regions_;
}

void SiPixelRawToDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("IncludeErrors", true);
  desc.add<bool>("UseQualityInfo", false);
  {
    desc.add<std::vector<int>>("ErrorList", std::vector<int>{29})
        ->setComment("## ErrorList: list of error codes used by tracking to invalidate modules");
  }
  {
    desc.add<std::vector<int>>("UserErrorList", std::vector<int>{40})
        ->setComment("## UserErrorList: list of error codes used by Pixel experts for investigation");
  }
  desc.add<edm::InputTag>("InputLabel", edm::InputTag("siPixelRawData"));
  {
    edm::ParameterSetDescription psd0;
    psd0.addOptional<std::vector<edm::InputTag>>("inputs");
    psd0.addOptional<std::vector<double>>("deltaPhi");
    psd0.addOptional<std::vector<double>>("maxZ");
    psd0.addOptional<edm::InputTag>("beamSpot");
    desc.add<edm::ParameterSetDescription>("Regions", psd0)
        ->setComment("## Empty Regions PSet means complete unpacking");
  }
  desc.add<bool>("UsePilotBlade", false)->setComment("##  Use pilot blades");
  desc.add<bool>("UsePhase1", false)->setComment("##  Use phase1");
  desc.add<std::string>("CablingMapLabel", "")->setComment("CablingMap label");  //Tav
  desc.addOptional<bool>("CheckPixelOrder");  // unused option, kept for back-compatibility
  descriptions.add("siPixelRawToDigi", desc);
}

// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
void SiPixelRawToDigi::produce(edm::Event& ev, const edm::EventSetup& es) {
  const uint32_t dummydetid = 0xffffffff;

  // initialize cabling map or update if necessary
  if (recordWatcher_.check(es)) {
    // cabling map, which maps online address (fed->link->ROC->local pixel) to offline (DetId->global pixel)
    edm::ESHandle<SiPixelFedCablingMap> cablingMap = es.getHandle(cablingMapToken_);
    fedIds_ = cablingMap->fedIds();
    cabling_ = cablingMap->cablingTree();
    LogDebug("map version:") << cabling_->version();
  }
  // initialize quality record or update if necessary
  if (qualityWatcher_.check(es) && useQuality_) {
    // quality info for dead pixel modules or ROCs
    edm::ESHandle<SiPixelQuality> qualityInfo = es.getHandle(siPixelQualityToken_);
    badPixelInfo_ = qualityInfo.product();
    if (!badPixelInfo_) {
      edm::LogError("SiPixelQualityNotPresent")
          << " Configured to use SiPixelQuality, but SiPixelQuality not present" << endl;
    }
  }

  edm::Handle<FEDRawDataCollection> buffers;
  ev.getByToken(fedRawDataCollectionToken_, buffers);

  // create product (digis & errors)
  auto collection = edm::DetSetVector<PixelDigi>();
  // collection->reserve(8*1024);
  auto errorcollection = edm::DetSetVector<SiPixelRawDataError>{};
  auto tkerror_detidcollection = DetIdCollection{};
  auto usererror_detidcollection = DetIdCollection{};
  auto disabled_channelcollection = edmNew::DetSetVector<PixelFEDChannel>{};

  PixelDataFormatter formatter(cabling_.get(), usePhase1_);  // for phase 1 & 0

  formatter.setErrorStatus(includeErrors_);

  if (useQuality_)
    formatter.setQualityStatus(useQuality_, badPixelInfo_);

  bool errorsInEvent = false;
  PixelDataFormatter::DetErrors nodeterrors;

  if (regions_) {
    regions_->run(ev, es);
    formatter.setModulesToUnpack(regions_->modulesToUnpack());
    LogDebug("SiPixelRawToDigi") << "region2unpack #feds: " << regions_->nFEDs();
    LogDebug("SiPixelRawToDigi") << "region2unpack #modules (BPIX,EPIX,total): " << regions_->nBarrelModules() << " "
                                 << regions_->nForwardModules() << " " << regions_->nModules();
  }

  for (auto aFed = fedIds_.begin(); aFed != fedIds_.end(); ++aFed) {
    int fedId = *aFed;

    if (!usePilotBlade_ && (fedId == 40))
      continue;  // skip pilot blade data

    if (regions_ && !regions_->mayUnpackFED(fedId))
      continue;

    LogDebug("SiPixelRawToDigi") << " PRODUCE DIGI FOR FED: " << fedId << endl;

    PixelDataFormatter::Errors errors;

    //get event data for this fed
    const FEDRawData& fedRawData = buffers->FEDData(fedId);

    //convert data to digi and strip off errors
    formatter.interpretRawData(errorsInEvent, fedId, fedRawData, collection, errors);

    //pack errors into collection
    if (includeErrors_) {
      formatter.unpackFEDErrors(errors,
                                tkerrorlist_,
                                usererrorlist_,
                                errorcollection,
                                tkerror_detidcollection,
                                usererror_detidcollection,
                                disabled_channelcollection,
                                nodeterrors);
    }  // if errors to be included in the event
  }    // loop on FED data to be unpacked

  if (includeErrors_) {
    edm::DetSet<SiPixelRawDataError>& errorDetSet = errorcollection.find_or_insert(dummydetid);
    errorDetSet.data = nodeterrors;
  }
  if (errorsInEvent)
    LogDebug("SiPixelRawToDigi") << "Error words were stored in this event";

  ev.emplace(siPixelDigiCollectionToken_, std::move(collection));
  if (includeErrors_) {
    ev.emplace(errorPutToken_, std::move(errorcollection));
    ev.emplace(tkErrorPutToken_, std::move(tkerror_detidcollection));
    ev.emplace(userErrorPutToken_, std::move(usererror_detidcollection));
    ev.emplace(disabledChannelPutToken_, std::move(disabled_channelcollection));
  }
}
// declare this as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiPixelRawToDigi);
