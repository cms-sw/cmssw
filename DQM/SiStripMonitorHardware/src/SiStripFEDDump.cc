#include <sstream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

//
// Class declaration
//

class SiStripFEDDumpPlugin : public edm::EDAnalyzer
{
 public:
  explicit SiStripFEDDumpPlugin(const edm::ParameterSet&);
  ~SiStripFEDDumpPlugin();
 private:
  virtual void beginJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  //tag of FEDRawData collection
  edm::InputTag rawDataTag_;
  //FED ID to dump
  unsigned int fedIdToDump_;
};


//
// Constructors and destructor
//

SiStripFEDDumpPlugin::SiStripFEDDumpPlugin(const edm::ParameterSet& iConfig)
  : rawDataTag_(iConfig.getUntrackedParameter<edm::InputTag>("RawDataTag",edm::InputTag("source",""))),
    fedIdToDump_(iConfig.getUntrackedParameter<unsigned int>("FEDID",50))
{
  if ( (fedIdToDump_ > FEDNumbering::MAXSiStripFEDID) || (fedIdToDump_ < FEDNumbering::MINSiStripFEDID) )
    edm::LogError("SiStripFEDDump") << "FED ID " << fedIdToDump_ << " is not valid. "
                                    << "SiStrip FED IDs are " << uint16_t(FEDNumbering::MINSiStripFEDID) << "-" << uint16_t(FEDNumbering::MAXSiStripFEDID);
}

SiStripFEDDumpPlugin::~SiStripFEDDumpPlugin()
{
}


//
// Member functions
//

// ------------ method called to for each event  ------------
void
SiStripFEDDumpPlugin::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //get raw data
  edm::Handle<FEDRawDataCollection> rawDataCollectionHandle;
  iEvent.getByLabel(rawDataTag_,rawDataCollectionHandle);
  const FEDRawDataCollection& rawDataCollection = *rawDataCollectionHandle;
  
  const FEDRawData& rawData = rawDataCollection.FEDData(fedIdToDump_);
  const sistrip::FEDBufferBase buffer(rawData.data(),rawData.size(),true);
  std::ostringstream os;
  os << buffer << std::endl;
  buffer.dump(os);
  edm::LogVerbatim("SiStripFEDDump") << os.str();
}

// ------------ method called once each job just before starting event loop  ------------
void 
SiStripFEDDumpPlugin::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiStripFEDDumpPlugin::endJob()
{
}

//
// Define as a plug-in
//

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripFEDDumpPlugin);
