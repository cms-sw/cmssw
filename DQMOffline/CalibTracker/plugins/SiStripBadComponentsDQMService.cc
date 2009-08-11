#include "DQMOffline/CalibTracker/plugins/SiStripBadComponentsDQMService.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include <string>
#include <sstream>
#include <cctype>
#include <time.h>
#include <boost/cstdint.hpp>

using namespace std;

SiStripBadComponentsDQMService::SiStripBadComponentsDQMService(const edm::ParameterSet& iConfig,const edm::ActivityRegistry& aReg):
  SiStripCondObjBuilderBase<SiStripBadStrip>::SiStripCondObjBuilderBase(iConfig),
  iConfig_(iConfig),
  fp_(iConfig.getUntrackedParameter<edm::FileInPath>("file",edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"))),
  notAlreadyRead_(true)
{
  obj_ = 0;
  edm::LogInfo("SiStripBadComponentsDQMService") <<  "[SiStripBadComponentsDQMService::SiStripBadComponentsDQMService]";
}

SiStripBadComponentsDQMService::~SiStripBadComponentsDQMService()
{
  edm::LogInfo("SiStripBadComponentsDQMService") <<  "[SiStripBadComponentsDQMService::~SiStripBadComponentsDQMService]";
}

void SiStripBadComponentsDQMService::getMetaDataString(std::stringstream& ss)
{
  ss << "Run " << getRunNumber() << endl;
  readBadComponents();
  obj_->printSummary(ss);
}

bool SiStripBadComponentsDQMService::checkForCompatibility(std::string ss)
{
  stringstream localString;
  getMetaDataString(localString);
  if( ss == localString.str() ) return false;

  return true;
}

void SiStripBadComponentsDQMService::readBadComponents()
{
  // Do this only if it was not already read
  if( notAlreadyRead_ ) {
    //*LOOP OVER THE LIST OF SUMMARY OBJECTS TO INSERT IN DB*//

      openRequestedFile();

    cout << "[readBadComponents]: opened requested file" << endl;

    obj_=new SiStripBadStrip();

    SiStripDetInfoFileReader reader(fp_.fullPath());

    dqmStore_->cd();

    string mdir = "MechanicalView";
    if (!goToDir(dqmStore_, mdir)) return;
    string mechanicalview_dir = dqmStore_->pwd();

    vector<string> subdet_folder;
    subdet_folder.push_back("TIB");
    subdet_folder.push_back("TOB");
    subdet_folder.push_back("TEC/side_1");
    subdet_folder.push_back("TEC/side_2");
    subdet_folder.push_back("TID/side_1");
    subdet_folder.push_back("TID/side_2");

    int nDetsTotal = 0;
    int nDetsWithErrorTotal = 0;
    for( vector<string>::const_iterator im = subdet_folder.begin(); im != subdet_folder.end(); ++im ) {
      string dname = mechanicalview_dir + "/" + (*im);
      if (!dqmStore_->dirExists(dname)) continue;

      dqmStore_->cd(dname);
      vector<string> module_folders;
      getModuleFolderList(dqmStore_, module_folders);
      int nDets = module_folders.size();

      int nDetsWithError = 0;
      string bad_module_folder = dname + "/" + "BadModuleList";
      if (dqmStore_->dirExists(bad_module_folder)) {
	std::vector<MonitorElement *> meVec = dqmStore_->getContents(bad_module_folder);
	for( std::vector<MonitorElement *>::const_iterator it = meVec.begin(); it != meVec.end(); ++it ) {
	  nDetsWithError++;
	  cout << (*it)->getName() <<  " " << (*it)->getIntValue() << endl;
	  uint32_t detId = boost::lexical_cast<uint32_t>((*it)->getName());
	  short flag = (*it)->getIntValue();

	  std::vector<unsigned int> theSiStripVector;

	  unsigned short firstBadStrip=0, NconsecutiveBadStrips=0;
	  unsigned int theBadStripRange;

	  // for(std::vector<uint32_t>::const_iterator is=BadApvList_.begin(); is!=BadApvList_.end(); ++is){

	  //   firstBadStrip=(*is)*128;
	  NconsecutiveBadStrips=reader.getNumberOfApvsAndStripLength(detId).first*128;

	  theBadStripRange = obj_->encode(firstBadStrip,NconsecutiveBadStrips,flag);

	  LogDebug("SiStripBadComponentsDQMService") << "detid " << detId << " \t"
						     << ", flag " << flag
						     << std::endl;

	  theSiStripVector.push_back(theBadStripRange);
	  // }

	  SiStripBadStrip::Range range(theSiStripVector.begin(),theSiStripVector.end());
	  if ( !obj_->put(detId,range) ) {
	    edm::LogError("SiStripBadFiberBuilder")<<"[SiStripBadFiberBuilder::analyze] detid already exists"<<std::endl;
	  }
	}
      }
      nDetsTotal += nDets;
      nDetsWithErrorTotal += nDetsWithError;        
    }
    dqmStore_->cd();
  }
}

void SiStripBadComponentsDQMService::openRequestedFile()
{
  dqmStore_ = edm::Service<DQMStore>().operator->();

  // ** FIXME ** // 
  dqmStore_->setVerbose(0); //add config param

  if( iConfig_.getParameter<bool>("accessDQMFile") ){
    
    std::string fileName = iConfig_.getUntrackedParameter<std::string>("FILE_NAME","");
    
    edm::LogInfo("SiStripBadComponentsDQMService") <<  "[SiStripBadComponentsDQMService::openRequestedFile] Accessing root File" << fileName;

    dqmStore_->open(fileName, false); 
  } else {
    edm::LogInfo("SiStripBadComponentsDQMService") <<  "[SiStripBadComponentsDQMService::openRequestedFile] Accessing dqmStore stream in Online Operation";
  }
}
 
uint32_t SiStripBadComponentsDQMService::getRunNumber() const {
  edm::LogInfo("SiStripBadComponentsDQMService") <<  "[SiStripBadComponentsDQMService::getRunNumber] " << iConfig_.getParameter<uint32_t>("RunNb");
  return iConfig_.getParameter<uint32_t>("RunNb");
}

bool SiStripBadComponentsDQMService::goToDir(DQMStore * dqm_store, string name)
{
  string currDir = dqm_store->pwd();
  string dirName = currDir.substr(currDir.find_last_of("/")+1);
  if (dirName.find(name) == 0) {
    return true;
  }
  vector<string> subDirVec = dqm_store->getSubdirs();
  for (vector<string>::const_iterator ic = subDirVec.begin();
       ic != subDirVec.end(); ic++) {
    dqm_store->cd(*ic);
    if (!goToDir(dqm_store, name))  dqm_store->goUp();
    else return true;
  }
  return false;
}

void SiStripBadComponentsDQMService::getModuleFolderList(DQMStore * dqm_store, vector<string>& mfolders)
{
  string currDir = dqm_store->pwd();
  if (currDir.find("module_") != string::npos)  {
    //    string mId = currDir.substr(currDir.find("module_")+7, 9);
    mfolders.push_back(currDir);
  } else {  
    vector<string> subdirs = dqm_store->getSubdirs();
    for( vector<string>::const_iterator it = subdirs.begin();
         it != subdirs.end(); ++it) {
      dqm_store->cd(*it);
      getModuleFolderList(dqm_store, mfolders);
      dqm_store->goUp();
    }
  }
}
