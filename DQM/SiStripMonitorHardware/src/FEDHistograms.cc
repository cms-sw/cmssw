#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"

#include "DQM/SiStripMonitorHardware/interface/FEDHistograms.hh"



FEDHistograms::FEDHistograms():
  dqm_(0),
  dqmPath_("")
{

}

FEDHistograms::~FEDHistograms()
{

}
  
void FEDHistograms::initialise(const edm::ParameterSet& iConfig,
			       std::ostringstream* pDebugStream
			       )
{
  getConfigForHistogram("DataPresent",iConfig,pDebugStream);
  getConfigForHistogram("AnyFEDErrors",iConfig,pDebugStream);
  getConfigForHistogram("AnyDAQProblems",iConfig,pDebugStream);
  getConfigForHistogram("AnyFEProblems",iConfig,pDebugStream);
  getConfigForHistogram("CorruptBuffers",iConfig,pDebugStream);
  getConfigForHistogram("BadChannelStatusBits",iConfig,pDebugStream);
  getConfigForHistogram("BadActiveChannelStatusBits",iConfig,pDebugStream);
  
  getConfigForHistogram("FEOverflows",iConfig,pDebugStream);
  getConfigForHistogram("FEMissing",iConfig,pDebugStream);
  getConfigForHistogram("BadMajorityAddresses",iConfig,pDebugStream);
  
  getConfigForHistogram("DataMissing",iConfig,pDebugStream);
  getConfigForHistogram("BadIDs",iConfig,pDebugStream);
  getConfigForHistogram("BadDAQPacket",iConfig,pDebugStream);
  getConfigForHistogram("InvalidBuffers",iConfig,pDebugStream);
  getConfigForHistogram("BadDAQCRCs",iConfig,pDebugStream);
  getConfigForHistogram("BadFEDCRCs",iConfig,pDebugStream);
  
  getConfigForHistogram("FEOverflowsDetailed",iConfig,pDebugStream);
  getConfigForHistogram("FEMissingDetailed",iConfig,pDebugStream);
  getConfigForHistogram("BadMajorityAddressesDetailed",iConfig,pDebugStream);
  getConfigForHistogram("BadAPVStatusBitsDetailed",iConfig,pDebugStream);
  getConfigForHistogram("APVErrorBitsDetailed",iConfig,pDebugStream);
  getConfigForHistogram("APVAddressErrorBitsDetailed",iConfig,pDebugStream);
  getConfigForHistogram("UnlockedBitsDetailed",iConfig,pDebugStream);
  getConfigForHistogram("OOSBitsDetailed",iConfig,pDebugStream);
  
  getConfigForHistogram("nFEDErrors",iConfig,pDebugStream);
  getConfigForHistogram("nFEDDAQProblems",iConfig,pDebugStream);
  getConfigForHistogram("nFEDsWithFEProblems",iConfig,pDebugStream);
  getConfigForHistogram("nFEDCorruptBuffers",iConfig,pDebugStream);
  getConfigForHistogram("nBadActiveChannelStatusBits",iConfig,pDebugStream);
  getConfigForHistogram("nFEDsWithFEOverflows",iConfig,pDebugStream);
  getConfigForHistogram("nFEDsWithMissingFEs",iConfig,pDebugStream);
  getConfigForHistogram("nFEDsWithFEBadMajorityAddresses",iConfig,pDebugStream);

}



void FEDHistograms::fillCountersHistogram(const FEDCounters & fedLevelCounters ){
  fillHistogram("nFEDErrors",fedLevelCounters.nFEDErrors);
  fillHistogram("nFEDDAQProblems",fedLevelCounters.nDAQProblems);
  fillHistogram("nFEDsWithFEProblems",fedLevelCounters.nFEDsWithFEProblems);
  fillHistogram("nFEDCorruptBuffers",fedLevelCounters.nCorruptBuffers);
  fillHistogram("nFEDsWithFEOverflows",fedLevelCounters.nFEDsWithFEOverflows);
  fillHistogram("nFEDsWithFEBadMajorityAddresses",fedLevelCounters.nFEDsWithFEBadMajorityAddresses);
  fillHistogram("nFEDsWithMissingFEs",fedLevelCounters.nFEDsWithMissingFEs);
  fillHistogram("nBadActiveChannelStatusBits",fedLevelCounters.nBadActiveChannels);
  
}


void FEDHistograms::fillHistogram(const std::string & histoName, 
				  const double value,
				  const int fedId
				  )
{
  std::string lPath = dqmPath_+"/"+histoName;
  if (fedId > 0){
    SiStripFedKey fedKey(fedId,0,0,0);
    std::stringstream fedIdStream;
    fedIdStream << fedId;
    lPath = fedKey.path()+histoName+fedIdStream.str();
  }
  MonitorElement *histogram = dqm_->get(lPath);
  if (histogram) histogram->Fill(value);
  //else {
    //std::cerr << "--- Histogram " << histoName << " not found ! " << std::endl;
    //exit(0);
  //}
}

void FEDHistograms::bookTopLevelHistograms(DQMStore* dqm,
					   const std::string & folderName)
{
  //get FED IDs
  const unsigned int siStripFedIdMin = FEDNumbering::MINSiStripFEDID;
  const unsigned int siStripFedIdMax = FEDNumbering::MAXSiStripFEDID;

  //get the pointer to the dqm object
  dqm_ = dqm;
  dqmPath_ = folderName;

  //book FED level histograms
  histosBooked_.resize(siStripFedIdMax+1,false);
  debugHistosBooked_.resize(siStripFedIdMax+1,false);


  //book histos
  dataPresent_ = bookHistogram("DataPresent","DataPresent",
                               "Number of events where the data from a FED is seen",
                               siStripFedIdMax-siStripFedIdMin+1,
                               siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  dataMissing_ = bookHistogram("DataMissing","DataMissing",
                               "Number of events where the data from a FED with cabled channels is missing",
                               siStripFedIdMax-siStripFedIdMin+1,
                               siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  anyFEDErrors_ = bookHistogram("AnyFEDErrors","AnyFEDErrors",
                             "Number of buffers with any FED error (excluding bad channel status bits, FE problems except overflows) per FED",
                             siStripFedIdMax-siStripFedIdMin+1,
                             siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  corruptBuffers_ = bookHistogram("CorruptBuffers","CorruptBuffers",
                                  "Number of corrupt FED buffers per FED",
                                  siStripFedIdMax-siStripFedIdMin+1,
                                  siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  invalidBuffers_ = bookHistogram("InvalidBuffers","InvalidBuffers",
                                  "Number of invalid FED buffers per FED",
                                  siStripFedIdMax-siStripFedIdMin+1,
                                  siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  anyDAQProblems_ = bookHistogram("AnyDAQProblems","AnyDAQProblems",
                                  "Number of buffers with any problems flagged in DAQ header (including CRC)",
                                  siStripFedIdMax-siStripFedIdMin+1,
                                  siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  badIDs_ = bookHistogram("BadIDs","BadIDs",
                          "Number of buffers with non-SiStrip source IDs in DAQ header",
                          siStripFedIdMax-siStripFedIdMin+1,
                          siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  badChannelStatusBits_ = bookHistogram("BadChannelStatusBits","BadChannelStatusBits",
                                        "Number of buffers with one or more enabled channel with bad status bits",
                                        siStripFedIdMax-siStripFedIdMin+1,
                                        siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  badActiveChannelStatusBits_ = bookHistogram("BadActiveChannelStatusBits","BadActiveChannelStatusBits",
                                              "Number of buffers with one or more active channel with bad status bits",
                                              siStripFedIdMax-siStripFedIdMin+1,
                                              siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  anyFEProblems_ = bookHistogram("AnyFEProblems","AnyFEProblems",
                                  "Number of buffers with any FE unit problems",
                                  siStripFedIdMax-siStripFedIdMin+1,
                                  siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  
  badDAQCRCs_ = bookHistogram("BadDAQCRCs","BadDAQCRCs",
                              "Number of buffers with bad CRCs from the DAQ",
                              siStripFedIdMax-siStripFedIdMin+1,
                              siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  badFEDCRCs_ = bookHistogram("BadFEDCRCs","BadFEDCRCs",
                              "Number of buffers with bad CRCs from the FED",
                              siStripFedIdMax-siStripFedIdMin+1,
                              siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  badDAQPacket_ = bookHistogram("BadDAQPacket","BadDAQPacket",
                               "Number of buffers with (non-CRC) problems flagged in DAQ header/trailer",
                               siStripFedIdMax-siStripFedIdMin+1,
                               siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  feOverflows_ = bookHistogram("FEOverflows","FEOverflows",
                               "Number of buffers with one or more FE overflow",
                               siStripFedIdMax-siStripFedIdMin+1,
                               siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  badMajorityAddresses_ = bookHistogram("BadMajorityAddresses","BadMajorityAddresses",
                                        "Number of buffers with one or more FE with a bad majority APV address",
                                        siStripFedIdMax-siStripFedIdMin+1,
                                        siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  feMissing_ = bookHistogram("FEMissing","FEMissing",
                             "Number of buffers with one or more FE unit payload missing",
                             siStripFedIdMax-siStripFedIdMin+1,
                             siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  
  nFEDErrors_ = bookHistogram("nFEDErrors","nFEDErrors",
                              "Number of FEDs with errors (exclusing channel status bits) per event","");
  nFEDDAQProblems_ = bookHistogram("nFEDDAQProblems","nFEDDAQProblems",
                                   "Number of FEDs with DAQ problems per event","");
  nFEDsWithFEProblems_ = bookHistogram("nFEDsWithFEProblems","nFEDsWithFEProblems",
                                       "Number of FEDs with FE problems per event","");
  nFEDCorruptBuffers_ = bookHistogram("nFEDCorruptBuffers","nFEDCorruptBuffers",
                                      "Number of FEDs with corrupt buffers per event","");
  nBadActiveChannelStatusBits_ = bookHistogram("nBadActiveChannelStatusBits","nBadActiveChannelStatusBits",
                                               "Number of active channels with bad status bits per event","");
  nFEDsWithFEOverflows_ = bookHistogram("nFEDsWithFEOverflows","nFEDsWithFEOverflows",
                                        "Number FEDs with FE units which overflowed per event","");
  nFEDsWithFEBadMajorityAddresses_ = bookHistogram("nFEDsWithFEBadMajorityAddresses","nFEDsWithFEBadMajorityAddresses",
                                                   "Number of FEDs with FE units with a bad majority address per event","");
  nFEDsWithMissingFEs_ = bookHistogram("nFEDsWithMissingFEs","nFEDsWithMissingFEs",
                                       "Number of FEDs with missing FE unit payloads per event","");

}

void FEDHistograms::bookFEDHistograms(unsigned int fedId,
				      bool fullDebugMode
				      )
{
  if (!histosBooked_[fedId]) {
    SiStripFedKey fedKey(fedId,0,0,0);
    dqm_->setCurrentFolder(fedKey.path());
    std::stringstream fedIdStream;
    fedIdStream << fedId;
    feOverflowDetailed_[fedId] = bookHistogram("FEOverflowsDetailed",
                                               "FEOverflowsForFED"+fedIdStream.str(),
                                               "FE overflows per FE unit for FED ID "+fedIdStream.str(),
                                               sistrip::FEUNITS_PER_FED,0,sistrip::FEUNITS_PER_FED,
                                               "FE-Index");
    badMajorityAddressDetailed_[fedId] = bookHistogram("BadMajorityAddressesDetailed",
                                                       "BadMajorityAddressesForFED"+fedIdStream.str(),
                                                       "Bad majority APV addresses per FE unit for FED ID "+fedIdStream.str(),
                                                       sistrip::FEUNITS_PER_FED,0,sistrip::FEUNITS_PER_FED,
                                                       "FE-Index");
    feMissingDetailed_[fedId] = bookHistogram("FEMissingDetailed",
                                              "FEMissingForFED"+fedIdStream.str(),
                                              "Buffers with FE Unit payload missing per FE unit for FED ID "+fedIdStream.str(),
                                              sistrip::FEUNITS_PER_FED,0,sistrip::FEUNITS_PER_FED,
                                              "FE-Index");
    badStatusBitsDetailed_[fedId] = bookHistogram("BadAPVStatusBitsDetailed",
                                                  "BadAPVStatusBitsForFED"+fedIdStream.str(),
                                                  "Bad apv status bits for FED ID "+fedIdStream.str(),
                                                  sistrip::APVS_PER_FED,0,sistrip::APVS_PER_FED,
                                                  "APV-Index");
     histosBooked_[fedId] = true;
  }
  if (fullDebugMode && !debugHistosBooked_[fedId]) {
    SiStripFedKey fedKey(fedId,0,0,0);
    dqm_->setCurrentFolder(fedKey.path());
    std::stringstream fedIdStream;
    fedIdStream << fedId;
    apvErrorDetailed_[fedId] = bookHistogram("APVErrorBitsDetailed",
                                             "APVErrorBitsForFED"+fedIdStream.str(),
                                             "APV errors for FED ID "+fedIdStream.str(),
                                             sistrip::APVS_PER_FED,0,sistrip::APVS_PER_FED,
                                             "APV-Index");
    apvAddressErrorDetailed_[fedId] = bookHistogram("APVAddressErrorBitsDetailed",
                                                    "APVAddressErrorBitsForFED"+fedIdStream.str(),
                                                    "Wrong APV address errors for FED ID "+fedIdStream.str(),
                                                    sistrip::APVS_PER_FED,0,sistrip::APVS_PER_FED,
                                                    "APV-Index");
    unlockedDetailed_[fedId] = bookHistogram("UnlockedBitsDetailed",
                                             "UnlockedBitsForFED"+fedIdStream.str(),
                                             "Unlocked channels for FED ID "+fedIdStream.str(),
                                             sistrip::FEDCH_PER_FED,0,sistrip::FEDCH_PER_FED,
                                             "Channel-Index");
    outOfSyncDetailed_[fedId] = bookHistogram("OOSBitsDetailed",
                                              "OOSBitsForFED"+fedIdStream.str(),
                                              "Out of sync channels for FED ID "+fedIdStream.str(),
                                              sistrip::FEDCH_PER_FED,0,sistrip::FEDCH_PER_FED,
                                              "Channel-Index");
    debugHistosBooked_[fedId] = true;
  }
}

void FEDHistograms::bookAllFEDHistograms()
{
  //get FED IDs
  const unsigned int siStripFedIdMin = FEDNumbering::MINSiStripFEDID;
  const unsigned int siStripFedIdMax = FEDNumbering::MAXSiStripFEDID;
  //book them
  for (unsigned int iFed = siStripFedIdMin; iFed <= siStripFedIdMax; iFed++) {
    bookFEDHistograms(iFed,true);
  }
}

void FEDHistograms::getConfigForHistogram(const std::string& configName, 
					  const edm::ParameterSet& psetContainingConfigPSet, 
					  std::ostringstream* pDebugStream
					  )
{

  HistogramConfig config;
  const std::string psetName = configName+std::string("HistogramConfig");
  if (psetContainingConfigPSet.exists(psetName)) {
    const edm::ParameterSet& pset = psetContainingConfigPSet.getUntrackedParameter<edm::ParameterSet>(psetName);
    config.enabled = (pset.exists("Enabled") ? pset.getUntrackedParameter<bool>("Enabled") : true);
    if (config.enabled) {
      config.nBins = (pset.exists("NBins") ? pset.getUntrackedParameter<unsigned int>("NBins") : 0);
      config.min = (pset.exists("Min") ? pset.getUntrackedParameter<double>("Min") : 0);
      config.max = (pset.exists("Max") ? pset.getUntrackedParameter<double>("Max") : 0);
      if (config.nBins) {
        if (pDebugStream) (*pDebugStream) << "\tHistogram: " << configName << "\tEnabled"
                                          << "\tNBins: " << config.nBins << "\tMin: " << config.min << "\tMax: " << config.max << std::endl;
      } else {
        if (pDebugStream) (*pDebugStream) << "\tHistogram: " << configName << "\tEnabled" << std::endl;
      }
    } else {
      config.enabled = false;
      config.nBins = 0;
      config.min = config.max = 0.;
      if (pDebugStream) (*pDebugStream) << "\tHistogram: " << configName << "\tDisabled" << std::endl;
    }
  } else {
    config.enabled = false;
    config.nBins = 0;
    config.min = config.max = 0.;
    if (pDebugStream) (*pDebugStream) << "\tHistogram: " << configName << "\tDisabled" << std::endl;
  }
  histogramConfig_[configName] = config;


}

MonitorElement* FEDHistograms::bookHistogram(const std::string& configName,
					     const std::string& name, 
					     const std::string& title,
					     const unsigned int nBins, 
					     const double min, 
					     const double max,
					     const std::string& xAxisTitle
					     )
{

  if (histogramConfig_[configName].enabled) {
    MonitorElement* histo = dqm_->book1D(name,title,nBins,min,max);
    histo->setAxisTitle(xAxisTitle,1);
    return histo;
  } else {
    return NULL;
  }

}

MonitorElement* FEDHistograms::bookHistogram(const std::string& configName,
					     const std::string& name, 
					     const std::string& title, 
					     const std::string& xAxisTitle
					     )
{
  return bookHistogram(configName,name,title,histogramConfig_[configName].nBins,histogramConfig_[configName].min,histogramConfig_[configName].max,xAxisTitle);

}
  
