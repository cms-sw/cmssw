#ifndef DQMOffline_SiStripBaseServiceFromDQM_SiStripBaseServiceFromDQM_H
#define DQMOffline_SiStripBaseServiceFromDQM_SiStripBaseServiceFromDQM_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "CondTools/SiStrip/interface/SiStripCondObjBuilderBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "TH1.h"

#include <string>
#include <memory>
#include <sstream>
#include <algorithm>
#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>



/**
  @class SiStripBaseServiceFromDQM
  @author M. De Mattia
  @ Base class for methods shared between all services reading from DQM and writing in the Database.
*/

template <class T>
class SiStripBaseServiceFromDQM : public SiStripCondObjBuilderBase<T>
{
 public:

  explicit SiStripBaseServiceFromDQM(const edm::ParameterSet&);
  virtual ~SiStripBaseServiceFromDQM();

  /// Used to fill the logDB
  virtual void getMetaDataString(std::stringstream& ss);
  /// Check is the transfer is needed
  virtual bool checkForCompatibility(std::string ss);

 protected:

  /// Uses DQMStore to access the DQM file
  void openRequestedFile();
  /// Uses DQM utilities to access the requested dir
  bool goToDir(const std::string & name);
  /// Fill the mfolders vector with the full list of directories for all the modules
  void getModuleFolderList(std::vector<std::string>& mfolders);
  /// Returns the run number from the cfg
  uint32_t getRunNumber() const;
  /**
   * Returns a pointer to the monitoring element corresponding to the given detId and name. <br>
   * The name convention for module histograms is NAME__det__DETID. The name provided
   * must be NAME, removing all the __det__DETID part. This latter part will be built
   * and attached internally using the provided detId.
   */
  MonitorElement * getModuleHistogram(const uint32_t detId, const std::string & name);

  DQMStore* dqmStore_;
  edm::ParameterSet iConfig_;
  boost::shared_ptr<SiStripFolderOrganizer> folderOrganizer_;

  // Simple functor to remove unneeded ME
  struct StringNotMatch
  {
    StringNotMatch(const std::string & name) :
      name_(name)
    {
    }
    bool operator()(const MonitorElement * ME) const
    {
      return( ME->getName().find(name_) == std::string::npos );
    }
  protected:
    std::string name_;
  };

};

template <class T>
SiStripBaseServiceFromDQM<T>::SiStripBaseServiceFromDQM(const edm::ParameterSet& iConfig):
  SiStripCondObjBuilderBase<T>::SiStripCondObjBuilderBase(iConfig),
  iConfig_(iConfig),
  folderOrganizer_(boost::shared_ptr<SiStripFolderOrganizer>(new SiStripFolderOrganizer))
{
  // Needed because this is a template inheriting from another template, so it cannot
  // access directly unnamed (independent from the template parameters) members.
  this->obj_ = 0;
}

template <class T>
SiStripBaseServiceFromDQM<T>::~SiStripBaseServiceFromDQM()
{
}

template <class T>
void SiStripBaseServiceFromDQM<T>::openRequestedFile()
{
  dqmStore_ = edm::Service<DQMStore>().operator->();

  // ** FIXME ** // 
  dqmStore_->setVerbose(0); //add config param

  if( iConfig_.getParameter<bool>("accessDQMFile") ){
    
    std::string fileName = iConfig_.getUntrackedParameter<std::string>("FILE_NAME","");
    
    edm::LogInfo("SiStripBaseServiceFromDQM") <<  "[SiStripBaseServiceFromDQM::openRequestedFile] Accessing root File" << fileName;

    dqmStore_->open(fileName, false); 
  } else {
    edm::LogInfo("SiStripBaseServiceFromDQM") <<  "[SiStripBaseServiceFromDQM::openRequestedFile] Accessing dqmStore stream in Online Operation";
  }
}

template <class T>
bool SiStripBaseServiceFromDQM<T>::goToDir(const std::string & name)
{
  std::string currDir = dqmStore_->pwd();
  std::string dirName = currDir.substr(currDir.find_last_of("/")+1);
  // Protection vs directories written with a trailing "/"
  if( dirName.length() == 0 ) {
    std::string currDirCopy(currDir, 0, currDir.length()-1);
    dirName = currDirCopy.substr(currDirCopy.find_last_of("/")+1);
  }
  if (dirName.find(name) == 0) {
    return true;
  }
  std::vector<std::string> subDirVec = dqmStore_->getSubdirs();
  for (std::vector<std::string>::const_iterator ic = subDirVec.begin();
       ic != subDirVec.end(); ic++) {
    dqmStore_->cd(*ic);
    if (!goToDir(name))  dqmStore_->goUp();
    else return true;
  }
  return false;
}

template <class T>
void SiStripBaseServiceFromDQM<T>::getModuleFolderList(std::vector<std::string>& mfolders)
{
  std::string currDir = dqmStore_->pwd();
  if (currDir.find("module_") != std::string::npos)  {
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

// template <class T>
// MonitorElement * SiStripBaseServiceFromDQM<T>::getModuleHistogram(const uint32_t detId, const std::string & name)
// {
//   // Take the full path to the histogram
//   std::string path;
//   folderOrganizer_->getFolderName(detId, path);
//   std::cout << "path = " << path << std::endl;
//   // build the name of the histogram
//   std::cout << "pwd = " << dqmStore_->pwd() << std::endl;
//   // std::string fullName(dqmStore_->pwd()+"/"+path+"/"+name+"__det__");
//   std::string fullName(path+"/"+name+"__det__");
//   fullName += boost::lexical_cast<std::string>(detId);

//   // ATTENTION: fixing the problem in the folderOrganizer
//   size_t firstSlash = fullName.find_first_of("/");
//   fullName = fullName.substr(firstSlash, fullName.size());
//   // fullName = dqmStore_->pwd() + "/SiStrip/Run summary" + fullName;
//   fullName = "SiStrip/Run summary" + fullName;

//   std::cout << "fullName = " << fullName << std::endl;

//   return dqmStore_->get(fullName);
// }

template <class T>
uint32_t SiStripBaseServiceFromDQM<T>::getRunNumber() const
{
  edm::LogInfo("SiStripBaseServiceFromDQM") <<  "[SiStripBaseServiceFromDQM::getRunNumber] " << iConfig_.getParameter<uint32_t>("RunNb");
  return iConfig_.getParameter<uint32_t>("RunNb");
}

template <class T>
void SiStripBaseServiceFromDQM<T>::getMetaDataString(std::stringstream& ss)
{
  std::cout << "SiStripPedestalsDQMService::getMetaDataString" << std::endl;
  ss << "Run " << getRunNumber() << std::endl;
}

template <class T>
bool SiStripBaseServiceFromDQM<T>::checkForCompatibility(std::string ss)
{
  std::stringstream localString;
  getMetaDataString(localString);
  if( ss == localString.str() ) return false;

  return true;
}

#endif //DQMOffline_SiStripBaseServiceFromDQM_SiStripBaseServiceFromDQM_H
