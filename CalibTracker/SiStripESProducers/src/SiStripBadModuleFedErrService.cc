
#include "CalibTracker/SiStripESProducers/interface/SiStripBadModuleFedErrService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

#include "boost/cstdint.hpp"
#include "boost/lexical_cast.hpp"

#include <cctype>
#include <time.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

SiStripBadModuleFedErrService::SiStripBadModuleFedErrService(const edm::ParameterSet& iConfig,const edm::ActivityRegistry& aReg):
  SiStripDepCondObjBuilderBase<SiStripBadStrip,SiStripFedCabling>::SiStripDepCondObjBuilderBase(iConfig),
  iConfig_(iConfig)
{
  edm::LogInfo("SiStripBadModuleFedErrService") <<  "[SiStripBadModuleFedErrService::SiStripBadModuleFedErrService]";
}

SiStripBadModuleFedErrService::~SiStripBadModuleFedErrService()
{
  edm::LogInfo("SiStripBadModuleFedErrService") <<  "[SiStripBadModuleFedErrService::~SiStripBadModuleFedErrService]";
}

void SiStripBadModuleFedErrService::getMetaDataString(std::stringstream& ss)
{
  ss << "Run " << getRunNumber() << std::endl;
  obj_->printSummary(ss);
}

bool SiStripBadModuleFedErrService::checkForCompatibility(std::string ss)
{
  std::stringstream localString;
  getMetaDataString(localString);
  if( ss == localString.str() ) return false;

  return true;
}

SiStripBadStrip* SiStripBadModuleFedErrService::readBadComponentsFromFed(const SiStripFedCabling *  cabling) {

  
  SiStripQuality* obj_  = new SiStripQuality();

  bool readFlag = iConfig_.getParameter<bool>("ReadFromFile");
  dqmStore_ = edm::Service<DQMStore>().operator->();

  
  if (readFlag && !openRequestedFile()) return obj_;
  
  dqmStore_->cd();
  
  std::string dname = "SiStrip/ReadoutView";
  std::string hpath = dname;
  hpath += "/FedIdVsApvId";
  if (dqmStore_->dirExists(dname)) {    
    MonitorElement* me = dqmStore_->get(hpath);
    if (me) {
      std::vector<std::pair<uint16_t, uint16_t>> channelList;
      getFedBadChannelList(me, channelList); 
      uint16_t fId_last = 9999;
      uint16_t fChan_last = 9999;
      std::map< uint32_t , std::set<int> > detectorMap;
      for (std::vector<std::pair<uint16_t, uint16_t>>::iterator it = channelList.begin(); it != channelList.end(); it++) {
	uint16_t fId = it->first;        

	uint16_t fChan = it->second/2;
        if (fId == fId_last && fChan == fChan_last) continue;

	FedChannelConnection channel = cabling->fedConnection(fId, fChan);
	const uint32_t detId =  channel.detId(); 
	const uint16_t ipair = channel.apvPairNumber();
        detectorMap[detId].insert(ipair);
      }
      for (std::map< uint32_t , std::set<int> >::iterator im =  detectorMap.begin(); im != detectorMap.end(); im++) {
        const uint32_t detId = im->first;
	std::set<int> pairs = im->second;		
        SiStripQuality::InputVector theSiStripVector;	  
	unsigned short firstBadStrip = 0;
	unsigned short fNconsecutiveBadStrips = 0;
	unsigned int theBadStripRange;
        int last_pair = -1;
	for (std::set<int>::iterator ip = pairs.begin(); ip != pairs.end(); ip++) {
          if (last_pair == -1) {
	    firstBadStrip = (*ip) * 128 * 2;
	    fNconsecutiveBadStrips = 128*2;
	  } else if ((*ip) - last_pair  > 1) {
	    theBadStripRange = obj_->encode(firstBadStrip,fNconsecutiveBadStrips);       
	    theSiStripVector.push_back(theBadStripRange);
	    firstBadStrip = (*ip) * 128 * 2;
	    fNconsecutiveBadStrips = 128*2;
	  } else { 
	    fNconsecutiveBadStrips += 128*2;
	  }
          last_pair = (*ip);
	}
	theBadStripRange = obj_->encode(firstBadStrip,fNconsecutiveBadStrips);       
	theSiStripVector.push_back(theBadStripRange);
	
	edm::LogInfo("SiStripBadModuleFedErrService") << " SiStripBadModuleFedErrService::readBadComponentsFromFed " 
						      << " detid " << detId 
						      << " firstBadStrip " << firstBadStrip 
						      << " NconsecutiveBadStrips " << fNconsecutiveBadStrips 
						      << " packed integer " << std::hex << theBadStripRange  << std::dec; 
	SiStripBadStrip::Range range(theSiStripVector.begin(),theSiStripVector.end());
	if ( !obj_->put(detId,range) ) {
	  edm::LogError("SiStripBadModuleFedErrService")<<"[SiStripBadModuleFedErrService::readBadComponentsFromFed] detid already exists"<<std::endl; 
	} 
      }
      obj_->cleanUp();
    }
  }
  return obj_;
}


bool SiStripBadModuleFedErrService::openRequestedFile() {
  
  std::string fileName = iConfig_.getParameter<std::string>("FileName");
    
  edm::LogInfo("SiStripBadModuleFedErrService") <<  "[SiStripBadModuleFedErrService::openRequestedFile] Accessing root File" << fileName;

  if (!dqmStore_->load(fileName, DQMStore::OpenRunDirs::StripRunDirs, true) ) {
    edm::LogError("SiStripBadModuleFedErrService")<<"[SiStripBadModuleFedErrService::openRequestedFile] Requested file " << fileName << "Can not be opened!! ";
    return false;
  } else return true;
}
 
uint32_t SiStripBadModuleFedErrService::getRunNumber() const {
  edm::LogInfo("SiStripBadModuleFedErrService") <<  "[SiStripBadModuleFedErrService::getRunNumber] " << iConfig_.getParameter<uint32_t>("RunNumber");
  return iConfig_.getParameter<uint32_t>("RunNumber");
}
void SiStripBadModuleFedErrService::getFedBadChannelList(MonitorElement* me, std::vector<std::pair<uint16_t,uint16_t> >& list) {
  float cutoff = iConfig_.getParameter<double>("BadStripCutoff");
  if (me->kind() == MonitorElement::DQM_KIND_TH2F) {
    TH2F* th2 = me->getTH2F();
    float entries = getProcessedEvents();
    if (!entries) entries = th2->GetBinContent(th2->GetMaximumBin()); 
    for (uint16_t i = 1; i < th2->GetNbinsY()+1; i++) { 
      for (uint16_t j = 1; j < th2->GetNbinsX()+1; j++) { 
        if (th2->GetBinContent(j,i) > cutoff * entries) {
	  edm::LogInfo("SiStripBadModuleFedErrService") << " [SiStripBadModuleFedErrService::getFedBadChannelList] :: FedId & Channel " << th2->GetYaxis()->GetBinLowEdge(i) <<   "  " << th2->GetXaxis()->GetBinLowEdge(j);
          list.push_back(std::pair<uint16_t, uint16_t>(th2->GetYaxis()->GetBinLowEdge(i), th2->GetXaxis()->GetBinLowEdge(j)));  
	}
      }
    }
  }
}
float SiStripBadModuleFedErrService::getProcessedEvents() {

  dqmStore_->cd();
  
  std::string dname = "SiStrip/ReadoutView";
  std::string hpath = dname;
  hpath += "/nTotalBadActiveChannels";
  if (dqmStore_->dirExists(dname)) {
    MonitorElement* me = dqmStore_->get(hpath);
    if (me) return (me->getEntries());
  } 
  return 0; 
}
