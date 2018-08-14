// -*- C++ -*-
//
// Package:    DQM/SiStripMonitorHardware
// Class:      SiStripFEDCheckPlugin
// 
/**\class SiStripFEDCheckPlugin SiStripFEDCheck.cc DQM/SiStripMonitorHardware/plugins/SiStripFEDCheck.cc

 Description: DQM source application to produce data integrety histograms for SiStrip data for use in HLT and Prompt reco
*/
//
// Original Author:  Nicholas Cripps
//         Created:  2008/09/16
//
//
#include <memory>

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"

#include "DQM/SiStripMonitorHardware/interface/FEDErrors.hh"

#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

//
// Class declaration
//

class SiStripFEDCheckPlugin : public DQMEDAnalyzer
{
 public:
  explicit SiStripFEDCheckPlugin(const edm::ParameterSet&);
  ~SiStripFEDCheckPlugin() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  bool hasFatalError(const FEDRawData& fedData, unsigned int fedId) const;
  bool hasNonFatalError(const FEDRawData& fedData, unsigned int fedId) const;
  void updateCabling(const edm::EventSetup& eventSetup);
  
  inline void fillPresent(unsigned int fedId, bool present);
  inline void fillFatalError(unsigned int fedId, bool fatalError);
  inline void fillNonFatalError(unsigned int fedId, float nonFatalError);
  
  void doUpdateIfNeeded();
  void updateHistograms();
  
  
  edm::InputTag rawDataTag_;
  edm::EDGetTokenT<FEDRawDataCollection> rawDataToken_;
  std::string dirName_;
  bool printDebug_;
  
  //Histograms
  bool doPLOTfedsPresent_, doPLOTfedFatalErrors_, doPLOTfedNonFatalErrors_;
  bool doPLOTnFEDinVsLS_, doPLOTnFEDinWdataVsLS_;
  MonitorElement* fedsPresent_;
  MonitorElement* fedFatalErrors_;
  MonitorElement* fedNonFatalErrors_;

  MonitorElement* nFEDinVsLS_;
  MonitorElement* nFEDinWdataVsLS_;
  
  //For histogram cache
  unsigned int updateFrequency_;//Update histograms with cached values every n events. If zero then fill normally every event
  //cache values
  std::vector<unsigned int> fedsPresentBinContents_;
  std::vector<unsigned int> fedFatalErrorBinContents_;
  std::vector<unsigned int> fedNonFatalErrorBinContents_;
  unsigned int eventCount_;//incremented by doUpdateIfNeeded()
  
  //Fine grained control of tests
  bool doPayloadChecks_, checkChannelLengths_, checkPacketCodes_, checkFELengths_, checkChannelStatusBits_, verbose_;
  
  //Cabling
  uint32_t cablingCacheId_;
  const SiStripFedCabling* cabling_;

  unsigned int siStripFedIdMin_;
  unsigned int siStripFedIdMax_;

  edm::ParameterSet conf_;
};


//
// Constructors and destructor
//

SiStripFEDCheckPlugin::SiStripFEDCheckPlugin(const edm::ParameterSet& iConfig)
  : rawDataTag_  (iConfig.getParameter<edm::InputTag>("RawDataTag"))
  , dirName_     (iConfig.getUntrackedParameter<std::string>("DirName","SiStrip/FEDIntegrity/"))
  , printDebug_  (iConfig.getUntrackedParameter<bool>("PrintDebugMessages",false))
  , doPLOTfedsPresent_       (iConfig.getParameter<bool>("doPLOTfedsPresent")      )
  , doPLOTfedFatalErrors_    (iConfig.getParameter<bool>("doPLOTfedFatalErrors")   )
  , doPLOTfedNonFatalErrors_ (iConfig.getParameter<bool>("doPLOTfedNonFatalErrors"))
  , doPLOTnFEDinVsLS_        (iConfig.getParameter<bool>("doPLOTnFEDinVsLS")       )
  , doPLOTnFEDinWdataVsLS_   (iConfig.getParameter<bool>("doPLOTnFEDinWdataVsLS")  )
  , fedsPresent_      (nullptr)
  , fedFatalErrors_   (nullptr)
  , fedNonFatalErrors_(nullptr)
  , nFEDinVsLS_       (nullptr)
  , nFEDinWdataVsLS_  (nullptr)
  , updateFrequency_(iConfig.getUntrackedParameter<unsigned int>("HistogramUpdateFrequency",0))
  , fedsPresentBinContents_     (FEDNumbering::MAXSiStripFEDID+1,0)
  , fedFatalErrorBinContents_   (FEDNumbering::MAXSiStripFEDID+1,0)
  , fedNonFatalErrorBinContents_(FEDNumbering::MAXSiStripFEDID+1,0)
  , eventCount_(0)
  , doPayloadChecks_       (iConfig.getUntrackedParameter<bool>("DoPayloadChecks",        true))
  , checkChannelLengths_   (iConfig.getUntrackedParameter<bool>("CheckChannelLengths",    true))
  , checkPacketCodes_      (iConfig.getUntrackedParameter<bool>("CheckChannelPacketCodes",true))
  , checkFELengths_        (iConfig.getUntrackedParameter<bool>("CheckFELengths",         true))
  , checkChannelStatusBits_(iConfig.getUntrackedParameter<bool>("CheckChannelStatus",     true))
  , verbose_               (iConfig.getUntrackedParameter<bool>("verbose",                false))
  , cablingCacheId_(0)
{
  rawDataToken_ = consumes<FEDRawDataCollection>(rawDataTag_);
  if (printDebug_ && !doPayloadChecks_ && (checkChannelLengths_ || checkPacketCodes_ || checkFELengths_)) {
    std::stringstream ss;
    ss << "Payload checks are disabled but individual payload checks have been enabled. The following payload checks will be skipped: ";
    if (checkChannelLengths_) ss << "Channel length check, ";
    if (checkPacketCodes_) ss << "Channel packet code check, ";
    if (checkChannelStatusBits_) ss << "Cabled channel status bits checks, ";
    if (checkFELengths_) ss << "FE Unit legnth check";
    edm::LogWarning("SiStripFEDCheck") << ss.str();
  }

  siStripFedIdMin_ = FEDNumbering::MINSiStripFEDID;
  siStripFedIdMax_ = FEDNumbering::MAXSiStripFEDID;
  
  conf_ = iConfig;
}

SiStripFEDCheckPlugin::~SiStripFEDCheckPlugin()
{
}


//
// Member functions
//

// ------------ method called to for each event  ------------
void
SiStripFEDCheckPlugin::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  //update cabling
  updateCabling(iSetup);
  
  //get raw data
  edm::Handle<FEDRawDataCollection> rawDataCollectionHandle;
  const bool gotData = iEvent.getByToken(rawDataToken_,rawDataCollectionHandle);
  if (verbose_) std::cout << "[SiStripFEDCheckPlugin::analyze] gotData ? " << (gotData ? "YES" : "NOPE") << std::endl;
  if (!gotData) {
    //module is required to silently do nothing when data is not present
    return;
  }
  const FEDRawDataCollection& rawDataCollection = *rawDataCollectionHandle;
  
  //FED errors
  FEDErrors lFedErrors;

  //loop over siStrip FED IDs
  size_t nFEDin = 0;
  size_t nFEDinWdata = 0;
  for (unsigned int fedId = siStripFedIdMin_; fedId <= siStripFedIdMax_; fedId++) {
    const FEDRawData& fedData = rawDataCollection.FEDData(fedId);

    //create an object to fill all errors
    //third param to false:save time by not initialising anything not used here
    lFedErrors.initialiseFED(fedId,cabling_,tTopo,false);

    //check data exists
    if (!fedData.size() || !fedData.data()) {
      fillPresent(fedId,false);
      continue;
    }
    if (verbose_) std::cout << "FED " << fedId;
    if (verbose_) std::cout << " fedData.size(): " << fedData.size();
    if (verbose_) std::cout << " fedData.data(): " << fedData.data() << std::endl;
    if (fedData.size()) nFEDin++;
    if (fedData.size() && fedData.data()) nFEDinWdata++;

    //fill buffer present histogram
    fillPresent(fedId,true);

    //check for fatal errors
    //no need for debug output
    bool hasFatalErrors = false;
    float rateNonFatal = 0;

    std::auto_ptr<const sistrip::FEDBuffer> buffer;

    if (!lFedErrors.fillFatalFEDErrors(fedData,0)) {
      hasFatalErrors = true;
    }
    else {
      //need to construct full object to go any further
      if (doPayloadChecks_ || checkChannelStatusBits_) {
	
	buffer.reset(new sistrip::FEDBuffer(fedData.data(),fedData.size(),true));
	if (doPayloadChecks_) {

	  bool channelLengthsOK = checkChannelLengths_ ? buffer->checkChannelLengthsMatchBufferLength() : true;
	  bool channelPacketCodesOK = checkPacketCodes_ ? buffer->checkChannelPacketCodes() : true;
	  bool feLengthsOK = checkFELengths_ ? buffer->checkFEUnitLengths() : true;
	  if ( !channelLengthsOK ||
	       !channelPacketCodesOK ||
	       !feLengthsOK ) {
	    hasFatalErrors = true;
	  }
	}
	if (checkChannelStatusBits_) rateNonFatal = lFedErrors.fillNonFatalFEDErrors(buffer.get(),cabling_);
      }
    }

    if (hasFatalErrors) {
      fillFatalError(fedId,true);
      if (printDebug_) {
	if (!buffer.get()) buffer.reset(new sistrip::FEDBuffer(fedData.data(),fedData.size(),true));
	edm::LogInfo("SiStripFEDCheck") << "Fatal error with FED ID " << fedId << ". Check summary: " 
					<< std::endl << buffer->checkSummary() << std::endl;
	std::stringstream ss;
	buffer->dump(ss);
	edm::LogInfo("SiStripFEDCheck") << ss.str();
      }
    }
    else {
      fillFatalError(fedId,false);
      //fill non-fatal errors histogram if there were no fatal errors
      fillNonFatalError(fedId,rateNonFatal);
      if (printDebug_ && rateNonFatal > 0) {
	if (!buffer.get()) buffer.reset(new sistrip::FEDBuffer(fedData.data(),fedData.size(),true));
	edm::LogInfo("SiStripFEDCheck") << "Non-fatal error with FED ID " << fedId 
					<< " for " << rateNonFatal << " of the channels. Check summary: " 
					<< std::endl << buffer->checkSummary() << std::endl;
	std::stringstream ss;
	buffer->dump(ss);
	edm::LogInfo("SiStripFEDCheck") << ss.str();
      }

    }
  }//loop over FED IDs
  if (verbose_) std::cout << "nFEDin: " << nFEDin << " nFEDinWdata: " << nFEDinWdata << std::endl;
  if (doPLOTnFEDinVsLS_)      nFEDinVsLS_      -> Fill(static_cast<double>(iEvent.id().luminosityBlock()),nFEDin);
  if (doPLOTnFEDinWdataVsLS_) nFEDinWdataVsLS_ -> Fill(static_cast<double>(iEvent.id().luminosityBlock()),nFEDinWdata);
  
  //update histograms if needed
  doUpdateIfNeeded();
}

// ------------ method called once each job just before starting event loop  ------------
void SiStripFEDCheckPlugin::bookHistograms(DQMStore::IBooker & ibooker , const edm::Run & run, const edm::EventSetup & eSetup)
{
  size_t nFED    = siStripFedIdMax_-siStripFedIdMin_+1;
  double xFEDmin = siStripFedIdMin_-0.5;
  double xFEDmax = siStripFedIdMax_+0.5;

  //get DQM store
  ibooker.setCurrentFolder(dirName_);
  //book histograms
  if (doPLOTfedsPresent_) {
    fedsPresent_ = ibooker.book1D("FEDEntries",
				  "Number of times FED buffer is present in data",
				  nFED, xFEDmin, xFEDmax);
    fedsPresent_->setAxisTitle("FED-ID",1);
  }

  if (doPLOTfedFatalErrors_) {
    fedFatalErrors_ = ibooker.book1D("FEDFatal",
				     "Number of fatal errors in FED buffer",
				     nFED, xFEDmin, xFEDmax);
    fedFatalErrors_->setAxisTitle("FED-ID",1);
  }

  if (doPLOTfedNonFatalErrors_) {
    fedNonFatalErrors_ = ibooker.book1D("FEDNonFatal",
					"Number of non fatal errors in FED buffer",
					nFED, xFEDmin, xFEDmax);
    fedNonFatalErrors_->setAxisTitle("FED-ID",1);
  }

  int    LSBin = conf_.getParameter<int>   ("LSBin");
  double LSMin = conf_.getParameter<double>("LSMin");
  double LSMax = conf_.getParameter<double>("LSMax");
  
  if (doPLOTnFEDinVsLS_) {
    nFEDinVsLS_ = ibooker.bookProfile("nFEDinVsLS",
				      "number of FED in Vs LS",
				      LSBin, LSMin,   LSMax,
				      nFED,  xFEDmin, xFEDmax);
    nFEDinVsLS_->setAxisTitle("LS",1);
    nFEDinVsLS_->setAxisTitle("FED-ID",2);
  }

  if (doPLOTnFEDinWdataVsLS_) {
    nFEDinWdataVsLS_ = ibooker.bookProfile("nFEDinWdataVsLS",
					   "number of FED in (with data) Vs LS",
					   LSBin, LSMin,   LSMax,
					   nFED,  xFEDmin, xFEDmax);
    nFEDinWdataVsLS_->setAxisTitle("LS",1);
    nFEDinWdataVsLS_->setAxisTitle("FED-ID",2);
  }
}

// ------------ method called once each run just after ending the event loop  ------------
void 
SiStripFEDCheckPlugin::endRun(edm::Run const&, edm::EventSetup const&)
{
  updateHistograms();
}

void SiStripFEDCheckPlugin::updateCabling(const edm::EventSetup& eventSetup)
{
  uint32_t currentCacheId = eventSetup.get<SiStripFedCablingRcd>().cacheIdentifier();
  if (cablingCacheId_ != currentCacheId) {
    edm::ESHandle<SiStripFedCabling> cablingHandle;
    eventSetup.get<SiStripFedCablingRcd>().get(cablingHandle);
    cabling_ = cablingHandle.product();
    cablingCacheId_ = currentCacheId;
  }
}

void SiStripFEDCheckPlugin::fillPresent(unsigned int fedId, bool present)
{
  if (present) {
    if (updateFrequency_) fedsPresentBinContents_[fedId]++;
    else 
      if (doPLOTfedsPresent_) fedsPresent_->Fill(fedId);
  }
}

void SiStripFEDCheckPlugin::fillFatalError(unsigned int fedId, bool fatalError)
{
  if (updateFrequency_) {
    if (fatalError) fedFatalErrorBinContents_[fedId]++;
  } else {
    //fedFatalErrors_->Fill( fatalError ? 1 : 0 );
    if (fatalError) 
      if (doPLOTfedFatalErrors_) fedFatalErrors_->Fill(fedId);
  }
}

void SiStripFEDCheckPlugin::fillNonFatalError(unsigned int fedId, float nonFatalError)
{
  if (updateFrequency_) {
    if (nonFatalError>0) fedNonFatalErrorBinContents_[fedId]++;//nonFatalError;
  } else {
    if (nonFatalError>0) 
      if (doPLOTfedNonFatalErrors_) fedNonFatalErrors_->Fill(fedId);
  }
}

void SiStripFEDCheckPlugin::doUpdateIfNeeded()
{
  eventCount_++;
  if (updateFrequency_ && (eventCount_%updateFrequency_ == 0)) {
    updateHistograms();
  }
}

void SiStripFEDCheckPlugin::updateHistograms()
{
  //if the cache is not being used then do nothing
  if (!updateFrequency_) return;
  unsigned int entriesFedsPresent = 0;
  unsigned int entriesFatalErrors = 0;
  unsigned int entriesNonFatalErrors = 0;
  for (unsigned int fedId = siStripFedIdMin_, bin = 1; fedId < siStripFedIdMax_+1; fedId++, bin++) {
    unsigned int fedsPresentBin = fedsPresentBinContents_[fedId];
    if (doPLOTfedsPresent_) fedsPresent_->getTH1()->SetBinContent(bin,fedsPresentBin);
    entriesFedsPresent += fedsPresentBin;
    unsigned int fedFatalErrorsBin = fedFatalErrorBinContents_[fedId];
    if (doPLOTfedFatalErrors_) fedFatalErrors_->getTH1()->SetBinContent(bin,fedFatalErrorsBin);
    entriesFatalErrors += fedFatalErrorsBin;
    unsigned int fedNonFatalErrorsBin = fedNonFatalErrorBinContents_[fedId];
    if (doPLOTfedNonFatalErrors_) fedNonFatalErrors_->getTH1()->SetBinContent(bin,fedNonFatalErrorsBin);
    entriesNonFatalErrors += fedNonFatalErrorsBin;
  }
  if (doPLOTfedsPresent_) fedsPresent_->getTH1()->SetEntries(entriesFedsPresent);
  if (doPLOTfedFatalErrors_) fedFatalErrors_->getTH1()->SetEntries(entriesFatalErrors);
  if (doPLOTfedNonFatalErrors_) fedNonFatalErrors_->getTH1()->SetEntries(entriesNonFatalErrors);
}
void
SiStripFEDCheckPlugin::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;

  // Directory to book histograms in
  desc.addUntracked<std::string>("DirName","SiStrip/FEDIntegrity/");
  // Raw data collection
  desc.add<edm::InputTag>("RawDataTag",edm::InputTag("source"));
  // Number of events to cache info before updating histograms
  // (set to zero to disable cache)
  // HistogramUpdateFrequency = cms.untracked.uint32(0),
  desc.addUntracked<unsigned int>("HistogramUpdateFrequency",1000);
  // Print info about errors buffer dumps to LogInfo(SiStripFEDCheck)
  desc.addUntracked<bool>("PrintDebugMessages",false);
  desc.add<bool>("doPLOTfedsPresent",      true);
  desc.add<bool>("doPLOTfedFatalErrors",   true);
  desc.add<bool>("doPLOTfedNonFatalErrors",true);
  desc.add<bool>("doPLOTnFEDinVsLS",       false);
  desc.add<bool>("doPLOTnFEDinWdataVsLS",  false);
  // Write the DQM store to a file (DQMStore.root) at the end of the run
  desc.addUntracked<bool>("WriteDQMStore",false);
  // Use to disable all payload (non-fatal) checks
  desc.addUntracked<bool>("DoPayloadChecks",true);
  // Use to disable check on channel lengths
  desc.addUntracked<bool>("CheckChannelLengths",true);
  // Use to disable check on channel packet codes
  desc.addUntracked<bool>("CheckChannelPacketCodes",true);
  // Use to disable check on FE unit lengths in full debug header
  desc.addUntracked<bool>("CheckFELengths",true);
  // Use to disable check on channel status bits
  desc.addUntracked<bool>("CheckChannelStatus",true);
  desc.add<int>         ("LSBin",5000);
  desc.add<double>      ("LSMin",   0.5);
  desc.add<double>      ("LSMax",5000.5);  

  descriptions.addDefault(desc);


}


//
// Define as a plug-in
//

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripFEDCheckPlugin);
