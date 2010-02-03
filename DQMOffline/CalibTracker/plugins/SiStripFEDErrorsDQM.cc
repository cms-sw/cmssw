#include "SiStripFEDErrorsDQM.h"

#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

using namespace std;
using namespace edm;

SiStripFEDErrorsDQM::SiStripFEDErrorsDQM(const edm::ParameterSet& iConfig) :
  SiStripBaseServiceFromDQM<SiStripBadStrip>::SiStripBaseServiceFromDQM(iConfig),
  iConfig_(iConfig),
  fp_(iConfig.getUntrackedParameter<edm::FileInPath>("file",edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"))),
  cablingCacheId_(0),
  threshold_(iConfig.getUntrackedParameter<double>("Threshold",0)),
  debug_(iConfig.getUntrackedParameter<unsigned int>("Debug",0))
{
  obj_ = 0;
  edm::LogInfo("SiStripFEDErrorsDQM") <<  "[SiStripFEDErrorsDQM::SiStripFEDErrorsDQM()]";
}

SiStripFEDErrorsDQM::~SiStripFEDErrorsDQM() {
  edm::LogInfo("SiStripFEDErrorsDQM") <<  "[SiStripFEDErrorsDQM::~SiStripFEDErrorsDQM]";
}

// ------------ method called to for each event  ------------
void SiStripFEDErrorsDQM::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //update cabling
  updateCabling(iSetup);

  if (readBadAPVs())
    {
      // Save the parameters to the db.
      edm::Service<cond::service::PoolDBOutputService> mydbservice;
      if( mydbservice.isAvailable() ){
	if( mydbservice->isNewTagRequest("SiStripBadStripRcd") ){
	  mydbservice->createNewIOV<SiStripBadStrip>(obj_, mydbservice->beginOfTime(),mydbservice->endOfTime(),"SiStripBadStripRcd");
	} else {
	  mydbservice->appendSinceTime<SiStripBadStrip>(obj_, mydbservice->currentTime(),"SiStripBadStripRcd");      
	}
      } else {
	edm::LogError("SiStripFEDErrorsDQM")<<"Service is unavailable"<<std::endl;
      }
    }

}

// ------------ method called once each job just before starting event loop  ------------
void 
SiStripFEDErrorsDQM::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiStripFEDErrorsDQM::endJob() {
}

void SiStripFEDErrorsDQM::updateCabling(const edm::EventSetup& iSetup)
{
  uint32_t currentCacheId = iSetup.get<SiStripFedCablingRcd>().cacheIdentifier();
  if (cablingCacheId_ != currentCacheId) {
    edm::ESHandle<SiStripFedCabling> cablingHandle;
    iSetup.get<SiStripFedCablingRcd>().get(cablingHandle);
    cabling_ = cablingHandle.product();
    cablingCacheId_ = currentCacheId;
  }
}

bool SiStripFEDErrorsDQM::readBadAPVs(){

  //std::cout << "[SiStripFEDErrorsDQM::readBadAPVs]" << std::endl;

  openRequestedFile();

  //std::cout << "[SiStripFEDErrorsDQM::readBadAPVs]: opened requested file" << std::endl;

  obj_=new SiStripBadStrip();

  SiStripDetInfoFileReader lReader(fp_.fullPath());

  std::ostringstream lPath;
  lPath << "Run " << getRunNumber() << "/SiStrip/Run summary/ReadoutView/";

  dqmStore_->setCurrentFolder(lPath.str());
  LogTrace("SiStripFEDErrorsDQM") << "[SiStripFEDErrorsDQM::readBadAPVs] Now in " << dqmStore_->pwd() << std::endl;

  std::string lBaseDir = dqmStore_->pwd();

  std::vector<std::pair<std::string,unsigned int> > lFedsFolder;
  //for FED errors, use summary folder and fedId=0
  //do not put a slash or "goToDir" won't work...
  lFedsFolder.push_back(std::pair<std::string,unsigned int>("FedMonitoringSummary",0));

  //for FE/channel/APV errors, they are written in a folder per FED, 
  //if there was at least one error.
  //So just loop on folders and see which ones exist.
  for (unsigned int ifed(FEDNumbering::MINSiStripFEDID); 
       ifed<= FEDNumbering::MAXSiStripFEDID;
       ifed++){//loop on FEDs

    std::ostringstream lFedDir;
    lFedDir << "FrontEndDriver" << ifed;
    if (!goToDir(lFedDir.str())) continue;
    //if (!dqmStore_->dirExists(lFedDir.str())) continue;
    else {
      if (debug_) LogTrace("SiStripFEDErrorsDQM") << "[SiStripFEDErrorsDQM::readBadAPVs] - Errors detected for FED " << ifed << std::endl;
      lFedsFolder.push_back(std::pair<std::string,unsigned int>(lFedDir.str(),ifed));
    }
    dqmStore_->goUp();
  }

  unsigned int nAPVsTotal = 0;
  //retrieve total number of APVs valid and connected from cabling:
  if (!cabling_) {
    edm::LogError("SiStripFEDErrorsDQM") << "[SiStripFEDErrorsDQM::readBadAPVs] cabling not filled, return false " << std::endl;
    return false;
  }
  const std::vector<uint16_t>& lFedVec = cabling_->feds();
  for (unsigned int iFed(0);iFed<lFedVec.size();iFed++){
    if (lFedVec.at(iFed) < sistrip::FED_ID_MIN || lFedVec.at(iFed) > sistrip::FED_ID_MAX) {
      edm::LogError("SiStripFEDErrorsDQM") << "[SiStripFEDErrorsDQM::readBadAPVs] Invalid fedid : " << lFedVec.at(iFed) << std::endl;
      continue;
    }
    const std::vector<FedChannelConnection>& lConnVec = cabling_->connections(lFedVec.at(iFed));
    for (unsigned int iConn(0); iConn<lConnVec.size();iConn++){
      const FedChannelConnection & lConnection = lConnVec.at(iConn);
      if (!lConnection.isConnected()) continue;
      unsigned int lDetid = lConnection.detId();
      if (!lDetid || lDetid == sistrip::invalid32_) continue;
      //2 APVs per channel....
      nAPVsTotal += 2;
    }
  }

  edm::LogInfo("SiStripFEDErrorsDQM") << "[SiStripFEDErrorsDQM::readBadAPVs] Total number of APVs found : " << nAPVsTotal << std::endl;

  unsigned int nAPVsWithErrorTotal = 0;
  unsigned int nFolders = 0;
  float lNorm = 0;

  
  for( std::vector<std::pair<std::string,unsigned int> >::const_iterator iFolder = lFedsFolder.begin(); 
       iFolder != lFedsFolder.end(); 
       ++iFolder ) {//loop on lFedsFolders
    std::string lDirName = lBaseDir + "/" + (*iFolder).first;
    unsigned int lFedId = (*iFolder).second;
    
    if (!goToDir((*iFolder).first)) continue;

    std::vector<MonitorElement *> lMeVec = dqmStore_->getContents(lDirName);
    
    if (nFolders == 0) {
      
      for( std::vector<MonitorElement *>::const_iterator iMe = lMeVec.begin(); 
	   iMe != lMeVec.end(); 
	   ++iMe ) {//loop on ME found in directory
      
	std::string lMeName = (*iMe)->getName() ;
	if (lMeName.find("nFEDErrors") != lMeName.npos){
	  lNorm = (*iMe)->getEntries();
	}
      }
      //if norm histo has not been found, no point in continuing....
      if (lNorm < 1) {
	edm::LogError("SiStripFEDErrorsDQM") << "[SiStripFEDErrorsDQM::readBadAPVs] nFEDErrors not found, norm is " << lNorm << std::endl;
	return false;
      }
    }

    unsigned int nAPVsWithError = 0;
        
    for( std::vector<MonitorElement *>::const_iterator iMe = lMeVec.begin(); 
	 iMe != lMeVec.end(); 
	 ++iMe ) {//loop on ME found in directory
      
      if ((*iMe)->getEntries() == 0) continue;
      std::string lMeName = (*iMe)->getName() ;
      
      bool lookForErrors = false;
      if (nFolders == 0) {
	//for the first element of lFedsFolder: this is FED errors
	lookForErrors = 
	  lMeName.find("DataMissing") != lMeName.npos ||
	  lMeName.find("AnyFEDErrors") != lMeName.npos || 
	  (lMeName.find("CorruptBuffer") != lMeName.npos && 
	   lMeName.find("nFED") == lMeName.npos);
      }
      else {
	//for the others, it is channel or FE errors.
	lookForErrors = 
	  lMeName.find("APVAddressError") != lMeName.npos ||
	  lMeName.find("APVError") != lMeName.npos ||
	  lMeName.find("BadMajorityAddresses") != lMeName.npos ||
	  lMeName.find("FEMissing") != lMeName.npos ||
	  lMeName.find("OOSBits") != lMeName.npos ||
	  lMeName.find("UnlockedBits") != lMeName.npos;
      }

      if (lookForErrors) readHistogram(*iMe,nAPVsWithError,lNorm,lFedId);

    }//loop on ME found in directory

    nAPVsWithErrorTotal += nAPVsWithError;      
    ++nFolders;

    dqmStore_->goUp();

  }//loop on lFedsFolders

  edm::LogInfo("SiStripFEDErrorsDQM") << "[SiStripFEDErrorsDQM::readBadAPVs] Total APVs with error found above threshold = " << nAPVsWithErrorTotal << std::endl;

  dqmStore_->cd();

  addErrors();

  return true;
}//method

void SiStripFEDErrorsDQM::readHistogram(MonitorElement* aMe,
                                        unsigned int & aCounter, 
                                        const float aNorm,
                                        const unsigned int aFedId)
{
  unsigned short lFlag = 0;
  std::string lMeName = aMe->getName();
  if (lMeName.find("DataMissing") != lMeName.npos) {
    lFlag = 0;
  }
  else if (lMeName.find("AnyFEDErrors") != lMeName.npos) {
    lFlag = 1;
  }
  else if (lMeName.find("CorruptBuffer") != lMeName.npos && 
	   lMeName.find("nFED") == lMeName.npos) {
    lFlag = 2;
  }
  else if (lMeName.find("FEMissing") != lMeName.npos) {
    lFlag = 3;
  }
  else if (lMeName.find("BadMajorityAddresses") != lMeName.npos) {
    lFlag = 4;
  }
  else if (lMeName.find("UnlockedBits") != lMeName.npos) {
    lFlag = 5;
  }
  else if (lMeName.find("OOSBits") != lMeName.npos) {
    lFlag = 6;
  }
  else if (lMeName.find("APVAddressError") != lMeName.npos) {
    lFlag = 7;
  }
  else if (lMeName.find("APVError") != lMeName.npos) {
    lFlag = 8;
  }
  else {
    edm::LogError("SiStripFEDErrorsDQM") << "[SiStripFEDErrorsDQM::readHistogramError] Shouldn't be here ..." << std::endl;
    return;
  }

  if (debug_) {
    LogTrace("SiStripFEDErrorsDQM") << "[SiStripFEDErrorsDQM::readHistogramError] Reading histo : " << lMeName << ", flag = " << lFlag << std::endl;
  }

  unsigned int lNBins = aMe->getNbinsX();
  int lBinShift = 0;
  bool lIsFedHist = false;
  bool lIsAPVHist = false;
  bool lIsFeHist = false;
  bool lIsChHist = false;

  if (lNBins > 200) {
    lBinShift = FEDNumbering::MINSiStripFEDID-1;//shift for FED ID from bin number
    lIsFedHist = true;
  }
  else {
    lBinShift = -1;//shift for channel/APV/FE id from bin number
    if (lNBins > 100) lIsAPVHist = true;
    else if (lNBins < 10) lIsFeHist = true;
    else lIsChHist = true;
  }

  if (debug_) { 
    LogTrace("SiStripFEDErrorsDQM") << "[SiStripFEDErrorsDQM::readHistogramError] lIsFedHist: " << lIsFedHist << std::endl
				    << "[SiStripFEDErrorsDQM::readHistogramError] lIsAPVHist: " << lIsAPVHist << std::endl
				    << "[SiStripFEDErrorsDQM::readHistogramError] lIsFeHist : " << lIsFeHist << std::endl
				    << "[SiStripFEDErrorsDQM::readHistogramError] lIsChHist : " << lIsChHist << std::endl;
  }

  for (unsigned int ibin(1); ibin<lNBins+1; ibin++){
    if (aMe->getBinContent(ibin)>0){
      float lStat = aMe->getBinContent(ibin)*1./aNorm;
      if (lStat <= threshold_) {
	if (debug_) LogTrace("SiStripFEDErrorsDQM") << "[SiStripFEDErrorsDQM::readHistogramError] ---- Below threshold : " << lStat << std::endl;
	continue;
      }
      if (lIsFedHist) {
	unsigned int lFedId = ibin+lBinShift;
	//loop on all enabled channels of this FED....
	for (unsigned int iChId = 0; 
	     iChId < sistrip::FEDCH_PER_FED; 
	     iChId++) {//loop on channels
	  const FedChannelConnection & lConnection = cabling_->connection(lFedId,iChId);
	  if (!lConnection.isConnected()) continue;
    	  addBadAPV(lConnection,0,lFlag,aCounter);
	}
      }
      else {
	if(lIsFeHist) {
	  unsigned int iFeId = ibin+lBinShift;
	  //loop on all enabled channels of this FE....
	  for (unsigned int iFeCh = 0; 
	     iFeCh < sistrip::FEDCH_PER_FEUNIT; 
	     iFeCh++) {//loop on channels
	    unsigned int iChId = sistrip::FEDCH_PER_FEUNIT*iFeId+iFeCh;
	    const FedChannelConnection & lConnection = cabling_->connection(aFedId,iChId);
	    if (!lConnection.isConnected()) continue;
	    addBadAPV(lConnection,0,lFlag,aCounter);
	  }
	}
	else {
	  unsigned int iChId = ibin+lBinShift;
	  if (lIsAPVHist) {
	    unsigned int iAPVid = iChId%2+1;
	    iChId = static_cast<unsigned int>(iChId/2.);
	    const FedChannelConnection & lConnection = cabling_->connection(aFedId,iChId);
	    addBadAPV(lConnection,iAPVid,lFlag,aCounter);

	  }//ifAPVhists
	  else {
	    const FedChannelConnection & lConnection = cabling_->connection(aFedId,iChId);
	    addBadAPV(lConnection,0,lFlag,aCounter);
	  }
	}//if not FE hist
      }//if not FED hist
    }//if entries in histo
  }//loop on bins
}//method readHistogram

void SiStripFEDErrorsDQM::addBadAPV(const FedChannelConnection & aConnection,
                                    const unsigned short aAPVNumber,
                                    const unsigned short aFlag,
                                    unsigned int & aCounter)
{
  if (!aConnection.isConnected()) {
    edm::LogWarning("SiStripFEDErrorsDQM") << "[SiStripFEDErrorsDQM::addBadAPV] Warning, incompatible cabling ! Channel is not connected, but entry found in histo ... " << std::endl;
    return;
  }
  unsigned int lDetid = aConnection.detId();
  if (!lDetid || lDetid == sistrip::invalid32_) {
    edm::LogWarning("SiStripFEDErrorsDQM") << "[SiStripFEDErrorsDQM::addBadAPV] Warning, DetId is invalid: " << lDetid << std::endl;
    return;
  }
  //unsigned short nChInModule = aConnection.nApvPairs();
  unsigned short lApvNum = 0;
  if (aAPVNumber < 2) {
    lApvNum = 2*aConnection.apvPairNumber();
    addBadStrips(aConnection,lDetid,lApvNum,aFlag,aCounter);
  }
  if (aAPVNumber == 0 || aAPVNumber == 2) {
    lApvNum = 2*aConnection.apvPairNumber()+1;
    addBadStrips(aConnection,lDetid,lApvNum,aFlag,aCounter);
  }
}


void SiStripFEDErrorsDQM::addBadStrips(const FedChannelConnection & aConnection,
                                       const unsigned int aDetId,
                                       const unsigned short aApvNum,
                                       const unsigned short aFlag,
                                       unsigned int & aCounter)
{
  // std::vector<unsigned int> lStripVector;
  unsigned int lBadStripRange;
  unsigned short lFirstBadStrip=aApvNum*128;
  unsigned short lConsecutiveBadStrips=128;

  lBadStripRange = obj_->encode(lFirstBadStrip,lConsecutiveBadStrips,aFlag);

  LogTrace("SiStripFEDErrorsDQM") << "[SiStripFEDErrorsDQM::addBadStrips] ---- Adding : detid " << aDetId
				  << " (FED " << aConnection.fedId() 
				  << ", Ch " << aConnection.fedCh () << ")"
				  << ", APV " << aApvNum
				  << ", flag " << aFlag
				  << std::endl;

  detIdErrors_[aDetId].push_back(lBadStripRange);

  // lStripVector.push_back(lBadStripRange);
  // SiStripBadStrip::Range lRange(lStripVector.begin(),lStripVector.end());
  // if ( !obj_->put(aDetId,lRange) ) {
  //   edm::LogError("SiStripFEDErrorsDQM")<<"[SiStripFEDErrorsDQM::addBadStrips] detid already exists." << std::endl;
  // }
  
  aCounter++;
}

void SiStripFEDErrorsDQM::addErrors()
{
  for( std::map<uint32_t, std::vector<uint32_t> >::const_iterator it = detIdErrors_.begin(); it != detIdErrors_.end(); ++it )
    {

      std::vector<uint32_t> lList = it->second;

      //map of first strip number and flag
      //purpose is to encode all existing flags into a unique one...
      std::map<unsigned short,unsigned short> lAPVMap;
      lAPVMap.clear();

      for (uint32_t iCh(0); iCh<lList.size(); iCh++) {
	SiStripBadStrip::data lData = obj_->decode(lList.at(iCh));
	unsigned short lFlag = 0;
	setFlagBit(lFlag,lData.flag);
      
	//std::cout << " -- Detid " << it->first << ", strip " << lData.firstStrip << ", flag " << lData.flag << std::endl;

	std::pair<std::map<unsigned short,unsigned short>::iterator,bool> lInsert = lAPVMap.insert(std::pair<unsigned short,unsigned short>(lData.firstStrip,lFlag));
	if (!lInsert.second) {
	  //std::cout << " ---- Adding bit : " << lData.flag << " to " << lInsert.first->second << ": ";
	  setFlagBit(lInsert.first->second,lData.flag);
	  //std::cout << lInsert.first->second << std::endl;
	}
      }

      //encode the new flag
      std::vector<unsigned int> lStripVector;
      unsigned short lConsecutiveBadStrips=128;

      for (std::map<unsigned short,unsigned short>::iterator lIter = lAPVMap.begin();
	   lIter != lAPVMap.end(); 
	   lIter++)
	{
	  lStripVector.push_back(obj_->encode(lIter->first,lConsecutiveBadStrips,lIter->second));
	}

      SiStripBadStrip::Range lRange(lStripVector.begin(),lStripVector.end());
      if ( !obj_->put(it->first,lRange) ) {
	edm::LogError("SiStripFEDErrorsDQM")<<"[SiStripFEDErrorsDQM::addBadStrips] detid already exists." << std::endl;
      }
    }
}

void SiStripFEDErrorsDQM::setFlagBit(unsigned short & aFlag, const unsigned short aBit)
{

  aFlag = aFlag | (0x1 << aBit) ;


}




//define this as a plug-in
// DEFINE_FWK_MODULE(SiStripFEDErrorsDQM);


