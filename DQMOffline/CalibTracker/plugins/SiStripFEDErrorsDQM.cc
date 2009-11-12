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

  readBadAPVs();

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

// ------------ method called once each job just before starting event loop  ------------
void 
SiStripFEDErrorsDQM::beginJob(const edm::EventSetup&)
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

void SiStripFEDErrorsDQM::readBadAPVs(){

  std::cout << "SiStripFEDErrorsDQM::readBadAPVs" << std::endl;

  openRequestedFile();

  std::cout << "[readBadComponents]: opened requested file" << std::endl;

  obj_=new SiStripBadStrip();

  SiStripDetInfoFileReader lReader(fp_.fullPath());

  dqmStore_->cd();

  std::string lBaseDir = "ReadoutView/";
  if (!goToDir(lBaseDir)) return;
  lBaseDir = dqmStore_->pwd();

  std::vector<std::pair<std::string,unsigned int> > lFedsFolder;
  //for FED errors, use summary folder and fedId=0
  lFedsFolder.push_back(std::pair<std::string,unsigned int>("FedMonitoringSummary/",0));

  //for FE/channel/APV errors, they are written in a folder per FED, 
  //if there was at least one error.
  //So just loop on folders and see which ones exist.
  for (unsigned int ifed(FEDNumbering::MINSiStripFEDID); 
       ifed<= FEDNumbering::MAXSiStripFEDID;
       ifed++){//loop on FEDs

    std::ostringstream lFedDir;
    lFedDir << "FrontEndDriver" << ifed;
    
    if (!dqmStore_->dirExists(lFedDir.str())) continue;
    else {
      if (debug_) std::cout << " - Errors detected for FED " << ifed << std::endl;
      lFedsFolder.push_back(std::pair<std::string,unsigned int>(lFedDir.str(),ifed));
    }
  }

  unsigned int nAPVsTotal = 0;
  //retrieve total number of APVs valid and connected from cabling:
  const std::vector<uint16_t>& lFedVec = cabling_->feds();
  for (unsigned int iFed(0);iFed<lFedVec.size();iFed++){
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

  if (debug_) std::cout << "Total number of APVs found : " << nAPVsTotal << std::endl;

  unsigned int nAPVsWithErrorTotal = 0;
  unsigned int nFolders = 0;
  float lNorm = 1;

  for( std::vector<std::pair<std::string,unsigned int> >::const_iterator iFolder = lFedsFolder.begin(); 
       iFolder != lFedsFolder.end(); 
       ++iFolder ) {//loop on lFedsFolders
    std::string lDirName = lBaseDir + "/" + (*iFolder).first;
    unsigned int lFedId = (*iFolder).second;
    
    if (!dqmStore_->dirExists(lDirName)) continue;

    dqmStore_->cd(lDirName);

    if (nFolders == 0) {
      std::string lNormHist = lDirName + "/nFEDErrors";
      lNorm = (dqmStore_->get(lNormHist))->getEntries();
    }

    unsigned int nAPVsWithError = 0;
    std::vector<MonitorElement *> lMeVec = dqmStore_->getContents(lDirName);
    
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
	  lMeName.find("CorruptBuffer") != lMeName.npos;
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

  }//loop on lFedsFolders

  if (debug_) std::cout << "Total APVs with error found above threshold = " << nAPVsWithErrorTotal << std::endl;

  dqmStore_->cd();
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
  else if (lMeName.find("CorruptBuffer") != lMeName.npos) {
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

  if (debug_) {
    std::cout << "Reading histo : " << lMeName << ", flag = " << lFlag << std::endl;
  }

  unsigned int lNBins = aMe->getNbinsX();
  int lBinShift = 0;
  bool lIsFedHist = false;
  bool lIsAPVHist = false;
  bool lIsFeHist = false;
  if (lNBins > 200) {
    lBinShift = FEDNumbering::MINSiStripFEDID-1;//shift for FED ID from bin number
    lIsFedHist = true;
  }
  else {
    lBinShift = -1;//shift for channel/APV/FE id from bin number
    if (lNBins > 100) lIsAPVHist = true;
    else if (lNBins < 10) lIsFeHist = true;
  }

  if (debug_) { 
    std::cout << "lIsFedHist: " << lIsFedHist << std::endl
	      << "lIsAPVHist: " << lIsAPVHist << std::endl
	      << "lIsFeHist: " << lIsFeHist << std::endl;
  }

  for (unsigned int ibin(1); ibin<lNBins+1; ibin++){
    if (aMe->getBinContent(ibin)>0){
      float lStat = aMe->getBinContent(ibin)*1./aNorm;
      if (lStat <= threshold_) continue;
      if (lIsFedHist) {
	unsigned int lFedId = ibin+lBinShift;
	//loop on all enabled channels of this FED....
	for (unsigned int iChId = 0; 
	     iChId < sistrip::FEDCH_PER_FED; 
	     iChId++) {//loop on channels
	  const FedChannelConnection & lConnection = cabling_->connection(lFedId,iChId);
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
  if (!aConnection.isConnected()) return;
  unsigned int lDetid = aConnection.detId();
  if (!lDetid || lDetid == sistrip::invalid32_) return;
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
  std::vector<unsigned int> lStripVector;
  unsigned int lBadStripRange;
  unsigned short lFirstBadStrip=aApvNum*128;
  unsigned short lConsecutiveBadStrips=128;

  lBadStripRange = obj_->encode(lFirstBadStrip,lConsecutiveBadStrips,aFlag);

  LogDebug("SiStripBadComponentsDQM") << "detid " << aDetId << " \t"
                                      << ", APV " << aApvNum
                                      << ", flag " << aFlag
                                      << std::endl;

  if (debug_) {
    std::cout << " ---- Adding : detid " << aDetId
	      << " (FED " << aConnection.fedId() 
	      << ", Ch " << aConnection.fedCh () << ")"
	      << ", APV " << aApvNum
	      << ", flag " << aFlag
	      << std::endl;
  }

  lStripVector.push_back(lBadStripRange);
  SiStripBadStrip::Range lRange(lStripVector.begin(),lStripVector.end());
  //if ( !obj_->put(aDetId,lRange) ) {
  //  edm::LogError("SiStripBadFiberBuilder")<<"[SiStripBadFiberBuilder::analyze] detid already exists." << std::endl;
  //}
  
  aCounter++;
}

//define this as a plug-in
// DEFINE_FWK_MODULE(SiStripFEDErrorsDQM);


