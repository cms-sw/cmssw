#include "DQMOffline/CalibTracker/plugins/SiStripDQMStoreReader.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

void SiStripDQMStoreReader::openRequestedFile()
{
  dqmStore_ = edm::Service<DQMStore>().operator->();

  // ** FIXME ** //
  dqmStore_->setVerbose(0); //add config param

  if ( m_accessDQMFile ) {
    edm::LogInfo("SiStripBaseServiceFromDQM") <<  "[SiStripBaseServiceFromDQM::openRequestedFile] Accessing root File" << m_fileName;

    dqmStore_->open(m_fileName, false);
  } else {
    edm::LogInfo("SiStripBaseServiceFromDQM") <<  "[SiStripBaseServiceFromDQM::openRequestedFile] Accessing dqmStore stream in Online Operation";
  }
}

bool SiStripDQMStoreReader::goToDir(const std::string & name)
{
  std::string currDir = dqmStore_->pwd();
  std::string dirName = currDir.substr(currDir.find_last_of("/")+1);
  // Protection vs directories written with a trailing "/"
  if( dirName.length() == 0 ) {
    std::string currDirCopy(currDir, 0, currDir.length()-1);
    dirName = currDirCopy.substr(currDirCopy.find_last_of("/")+1);
  }
  if ( dirName.find(name) == 0 ) {
    return true;
  }
  std::vector<std::string> subDirVec = dqmStore_->getSubdirs();
  for ( std::vector<std::string>::const_iterator ic = subDirVec.begin();
        ic != subDirVec.end(); ++ic ) {
    dqmStore_->cd(*ic);
    if ( ! goToDir(name) ) dqmStore_->goUp();
    else return true;
  }
  return false;
}

void SiStripDQMStoreReader::getModuleFolderList(std::vector<std::string>& mfolders)
{
  std::string currDir = dqmStore_->pwd();
  if ( currDir.find("module_") != std::string::npos )  {
    //    std::string mId = currDir.substr(currDir.find("module_")+7, 9);
    mfolders.push_back(currDir);
  } else {
    std::vector<std::string> subdirs = dqmStore_->getSubdirs();
    for( std::vector<std::string>::const_iterator it = subdirs.begin();
         it != subdirs.end(); ++it) {
      dqmStore_->cd(*it);
      getModuleFolderList(mfolders);
      dqmStore_->goUp();
    }
  }
}

