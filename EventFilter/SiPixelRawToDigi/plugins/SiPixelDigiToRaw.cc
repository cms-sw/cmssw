#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/EDProducer.h"
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

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"

#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDCabling.h"

#include "TH1D.h"
#include "TFile.h"

class SiPixelDigiToRaw final : public edm::EDProducer {
public:

  /// ctor
  explicit SiPixelDigiToRaw( const edm::ParameterSet& );


  /// get data, convert to raw event, attach again to Event
  void produce( edm::Event&, const edm::EventSetup& ) override;

  // Fill parameters descriptions
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:

  std::unique_ptr<SiPixelFedCablingTree> cablingTree_;
  std::unique_ptr<SiPixelFrameReverter> frameReverter_;
  edm::ESWatcher<SiPixelFedCablingMapRcd> recordWatcher;
  const edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> tPixelDigi; 
  const edm::EDPutTokenT<FEDRawDataCollection> putToken_;
  const bool usePilotBlade = false;  // I am not yet sure we need it here?
  const bool usePhase1;
};

using namespace std;

SiPixelDigiToRaw::SiPixelDigiToRaw( const edm::ParameterSet& pset ) :
  frameReverter_(),
  tPixelDigi{ consumes<edm::DetSetVector<PixelDigi> >(pset.getParameter<edm::InputTag>("InputLabel")) },
  putToken_{produces<FEDRawDataCollection>()},
  usePhase1{ pset.getParameter<bool> ("UsePhase1") }
{


  // Define EDProduct type

  if(usePhase1) edm::LogInfo("SiPixelRawToDigi")  << " Use pilot blade data (FED 40)";

}

// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
void SiPixelDigiToRaw::produce( edm::Event& ev,
                              const edm::EventSetup& es)
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

  if (recordWatcher.check( es )) {
    edm::ESHandle<SiPixelFedCablingMap> cablingMap;
    es.get<SiPixelFedCablingMapRcd>().get( cablingMap );
    cablingTree_= cablingMap->cablingTree();
    frameReverter_ = std::make_unique<SiPixelFrameReverter>( es, cablingMap.product() );
  }

  LogDebug("SiPixelDigiToRaw") << cablingTree_->version();

  //PixelDataFormatter formatter(cablingTree_.get());
  PixelDataFormatter formatter(cablingTree_.get(), usePhase1);

  formatter.passFrameReverter(frameReverter_.get());

  // create product (raw data)
  FEDRawDataCollection buffers;

  // convert data to raw
  formatter.formatRawData( ev.id().event(), rawdata, digis );

  // pack raw data into collection
  for (auto const* fed: cablingTree_->fedList()) {
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
  descriptions.add("siPixelRawData",desc);
}

DEFINE_FWK_MODULE(SiPixelDigiToRaw);
