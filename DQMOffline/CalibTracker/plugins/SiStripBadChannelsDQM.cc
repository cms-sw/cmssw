#include "SiStripBadChannelsDQM.h"

using namespace std;
using namespace edm;

SiStripBadChannelsDQM::SiStripBadChannelsDQM(const edm::ParameterSet& iConfig) :
  SiStripBaseServiceFromDQM<SiStripBadStrip>::SiStripBaseServiceFromDQM(iConfig),
  iConfig_(iConfig),
  fp_(iConfig.getUntrackedParameter<edm::FileInPath>("file",edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"))),
  threshold_(iConfig.getUntrackedParameter<double>("Threshold",0)),
  debug_(iConfig.getUntrackedParameter<unsigned int>("Debug",0))
{
  obj_ = 0;
  edm::LogInfo("SiStripBadChannelsDQM") <<  "[SiStripBadChannelsDQM::SiStripBadChannelsDQM()]";
  tkDetMap_=edm::Service<TkDetMap>().operator->();
}

SiStripBadChannelsDQM::~SiStripBadChannelsDQM() {
  edm::LogInfo("SiStripBadChannelsDQM") <<  "[SiStripBadChannelsDQM::~SiStripBadChannelsDQM]";
}

// ------------ method called to for each event  ------------
void SiStripBadChannelsDQM::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  if (readBadChannels())
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
	edm::LogError("SiStripBadChannelsDQM")<<"Service is unavailable"<<std::endl;
      }
    }
  
}

// ------------ method called once each job just before starting event loop  ------------
void 
SiStripBadChannelsDQM::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiStripBadChannelsDQM::endJob() {
}

bool SiStripBadChannelsDQM::readBadChannels(){
  
  openRequestedFile();
  obj_=new SiStripBadStrip();
  
  SiStripDetInfoFileReader lReader(fp_.fullPath());
  
  const std::map<uint32_t, SiStripDetInfoFileReader::DetInfo> DetInfos = lReader.getAllData();
  
  std::ostringstream lPath;
  lPath << "Run " << getRunNumber() << "/SiStrip/Run summary/MechanicalView/";
  
  dqmStore_->setCurrentFolder(lPath.str());
  LogTrace("SiStripBadChannelsDQM") << "[SiStripBadChannelsDQM::readBadChannels] Now in " << dqmStore_->pwd() << std::endl;
  
  // Check the Bad Channels in each sub-detector
  std::vector<std::string> subDetFolder;
  subDetFolder.push_back("TIB");
  subDetFolder.push_back("TOB");
  subDetFolder.push_back("TEC/side_1");
  subDetFolder.push_back("TEC/side_2");
  subDetFolder.push_back("TID/side_1");
  subDetFolder.push_back("TID/side_2");
  
  unsigned int nBadTotal = 0;
  unsigned int nCheckedTotal = 0;
  
  for( std::vector<std::string>::const_iterator iSubDet = subDetFolder.begin(); iSubDet != subDetFolder.end(); ++iSubDet ) {
    std::ostringstream lSubDetDir;
    lSubDetDir << lPath.str() << (*iSubDet);
    dqmStore_->cd(lSubDetDir.str());
    int nElements = 0;
    std::ostringstream elementName;
    std::string lSubDetName = (*iSubDet);
    
    if 		(lSubDetName.find("TIB") != lSubDetName.npos) { nElements = 4; elementName << "layer";}
    else if 	(lSubDetName.find("TOB") != lSubDetName.npos) { nElements = 6; elementName << "layer";}
    else if 	(lSubDetName.find("TEC") != lSubDetName.npos) { nElements = 9; elementName << "wheel";}
    else if 	(lSubDetName.find("TID") != lSubDetName.npos) { nElements = 3; elementName << "wheel";}
    
    for (int iElem = 1; iElem < nElements+1; iElem++){
      
      std::ostringstream lElemDir;
      lElemDir << lSubDetDir.str() << "/" << elementName.str() << "_" << iElem ;
      dqmStore_->cd(lElemDir.str());
      std::vector<MonitorElement *> lMeVec = dqmStore_->getContents(lElemDir.str());
      
      for( std::vector<MonitorElement *>::const_iterator iMe = lMeVec.begin(); iMe != lMeVec.end(); ++iMe ) {
	
	std::string lMeName = (*iMe)->getName() ;
	if (lMeName.find("TkHMap_FractionOfBadChannels") != lMeName.npos){
	  int nBinsX = (*iMe)->getNbinsX();
	  int nBinsY = (*iMe)->getNbinsY();
	  for (int ibinx = 1; ibinx<nBinsX+1; ibinx++){
	    for (int ibiny = 1; ibiny<nBinsY+1; ibiny++){
	      nCheckedTotal++;
	      if ((*iMe)->getBinContent(ibinx, ibiny) >= threshold_) {
		std::string lname;
   		lname = lMeName.substr(lMeName.find("Channels_")+9);
	        uint32_t detId = tkDetMap_->getDetFromBin(lname, ibinx, ibiny);  
		unsigned short lFlag = (*iMe)->getBinContent(ibinx, ibiny);
		
		//cout << lSubDetName << "/" << elementName.str() << "_" << iElem<< ": (x " << ibinx << ", y " << ibiny << ") "
		  //   << (*iMe)->getBinContent(ibinx, ibiny) << " >> detId:  " << detId << endl;
	        
		std::map<uint32_t, SiStripDetInfoFileReader::DetInfo>::const_iterator detInfoIt = DetInfos.find(detId);
		if( detInfoIt != DetInfos.end() ) {
		  //cout << detInfoIt->second.nApvs << endl;
		  // Temporarily the value (*iMe)->getBinContent(ibinx, ibiny) is stored as flag
		  addBadStrips(detId, detInfoIt->second.nApvs, lFlag);
		}	
		nBadTotal++;
	      } 
	    }
	  } // bins y
	} // bins x
      } // reading the DQM histogram
    } // loop over layers and wheels
  } // loop over the subdetector

  //cout << "Total bad channels: " << nBadTotal << ", over " << nCheckedTotal << " checked " << endl;
  
  addErrors();
  
  dqmStore_->cd();
  
  if (nBadTotal) return true;
  else return false;

}

void SiStripBadChannelsDQM::addBadStrips(const unsigned int aDetId,
					 const unsigned short aApvNum,
					 const unsigned short aFlag)
{
  unsigned int lBadStripRange;
  unsigned short lFirstBadStrip=0;                                                                                                                                               
  unsigned short lConsecutiveBadStrips=aApvNum*128;

  lBadStripRange = obj_->encode(lFirstBadStrip,lConsecutiveBadStrips,aFlag);
  
  LogTrace("SiStripBadChannelsDQM") << "[SiStripBadChannelsDQM::addBadStrips] ---- Adding : detid " << aDetId
				    << ", APV " << aApvNum
				    << ", flag " << aFlag
				    << std::endl;
  
  detIdErrors_[aDetId].push_back(lBadStripRange); 
}

void SiStripBadChannelsDQM::addErrors()
{
  for( std::map<uint32_t, std::vector<uint32_t> >::const_iterator it = detIdErrors_.begin(); it != detIdErrors_.end(); ++it ) {
    SiStripBadStrip::Range lRange(it->second.begin(),it->second.end());
    if ( !obj_->put(it->first,lRange) ) {
      edm::LogError("SiStripFEDErrorsDQM")<<"[SiStripFEDErrorsDQM::addBadStrips] detid already exists." << std::endl;
    }
  }
}

//define this as a plug-in
// DEFINE_FWK_MODULE(SiStripBadChannelsDQM);


