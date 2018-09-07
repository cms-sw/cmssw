#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/CPUTimer.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameReverter.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"

#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDCabling.h"

#include <atomic>
#include <memory>

namespace sipixeldigitoraw {
  struct Cache {
    std::unique_ptr<SiPixelFedCablingTree> cablingTree_;
    std::unique_ptr<SiPixelFrameReverter> frameReverter_;
  };
}

namespace pr = sipixeldigitoraw;

class SiPixelDigiToRaw final : public edm::global::EDProducer<edm::LuminosityBlockCache<pr::Cache>> {
public:

  /// ctor
  explicit SiPixelDigiToRaw( const edm::ParameterSet& );


  /// get data, convert to raw event, attach again to Event
  void produce( edm::StreamID, edm::Event&, const edm::EventSetup& ) const final;

  std::shared_ptr<pr::Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const&, 
                                                        edm::EventSetup const& iES) const final;

  void globalEndLuminosityBlock(edm::LuminosityBlock const&,
                                edm::EventSetup const& iES) const final {}
  
  // Fill parameters descriptions
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:

  mutable std::atomic_flag lock_{ ATOMIC_FLAG_INIT };
  CMS_THREAD_GUARD(lock_) mutable edm::ESWatcher<SiPixelFedCablingMapRcd> recordWatcher;
  CMS_THREAD_GUARD(lock_) mutable std::shared_ptr<pr::Cache> previousCache_;
  const edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> tPixelDigi; 
  const edm::EDPutTokenT<FEDRawDataCollection> putToken_;
  const bool usePilotBlade = false;  // I am not yet sure we need it here?
  const bool usePhase1;
};

using namespace std;

SiPixelDigiToRaw::SiPixelDigiToRaw( const edm::ParameterSet& pset ) :
  tPixelDigi{ consumes<edm::DetSetVector<PixelDigi> >(pset.getParameter<edm::InputTag>("InputLabel")) },
  putToken_{produces<FEDRawDataCollection>()},
  usePhase1{ pset.getParameter<bool> ("UsePhase1") }
{


  // Define EDProduct type

  if(usePhase1) edm::LogInfo("SiPixelRawToDigi")  << " Use pilot blade data (FED 40)";

}

// -----------------------------------------------------------------------------
std::shared_ptr<pr::Cache> 
SiPixelDigiToRaw::globalBeginLuminosityBlock(edm::LuminosityBlock const&, 
                                             edm::EventSetup const& es) const {
  while(lock_.test_and_set(std::memory_order_acquire)); //spin
  auto rel = [](std::atomic_flag* f) { f->clear(std::memory_order_release); };
  std::unique_ptr<std::atomic_flag, decltype(rel)> guard(&lock_, rel);

  if (recordWatcher.check( es )) {
    edm::ESHandle<SiPixelFedCablingMap> cablingMap;
    es.get<SiPixelFedCablingMapRcd>().get( cablingMap );
    previousCache_ = std::make_shared<pr::Cache>();
    previousCache_->cablingTree_= cablingMap->cablingTree();
    previousCache_->frameReverter_ = std::make_unique<SiPixelFrameReverter>( es, cablingMap.product() );
  }
  return previousCache_;
}



// -----------------------------------------------------------------------------
void SiPixelDigiToRaw::produce( edm::StreamID, edm::Event& ev,
                                const edm::EventSetup& es) const
{
  using namespace sipixelobjects;

  edm::Handle< edm::DetSetVector<PixelDigi> > digiCollection;
  ev.getByToken( tPixelDigi, digiCollection);

  PixelDataFormatter::RawData rawdata;
  PixelDataFormatter::Digis digis;

  int digiCounter = 0; 
  for (auto const& di : *digiCollection) {
    digiCounter += (di.data).size(); 
    digis[ di.id] = di.data;
  }

  auto cache =  luminosityBlockCache(ev.getLuminosityBlock().index());


  LogDebug("SiPixelDigiToRaw") << cache->cablingTree_->version();

  //PixelDataFormatter formatter(cablingTree_.get());
  PixelDataFormatter formatter(cache->cablingTree_.get(), usePhase1);

  formatter.passFrameReverter(cache->frameReverter_.get());

  // create product (raw data)
  FEDRawDataCollection buffers;

  // convert data to raw
  formatter.formatRawData( ev.id().event(), rawdata, digis );

  // pack raw data into collection
  for (auto const* fed: cache->cablingTree_->fedList()) {
    LogDebug("SiPixelDigiToRaw")<<" PRODUCE DATA FOR FED_id: " << fed->id();
    FEDRawData& fedRawData = buffers.FEDData( fed->id() );
    PixelDataFormatter::RawData::iterator fedbuffer = rawdata.find( fed->id() );
    if( fedbuffer != rawdata.end() ) fedRawData = fedbuffer->second;
    LogDebug("SiPixelDigiToRaw")<<"size of data in fedRawData: "<<fedRawData.size();
  }

  LogDebug("SiPixelDigiToRaw").log([&](auto &l) {

      l << "Words/Digis this ev: "<<digiCounter<<"(fm:"<<formatter.nDigis()<<")/"
        <<formatter.nWords();
    });
  ev.emplace(putToken_, std::move(buffers));
  
}

// -----------------------------------------------------------------------------
void SiPixelDigiToRaw::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputLabel");
  desc.add<bool>("UsePhase1", false);
  desc.addUntracked<bool>("Timing", false)->setComment("deprecated");
  descriptions.add("siPixelRawData",  desc);
}

// declare this as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiPixelDigiToRaw);
