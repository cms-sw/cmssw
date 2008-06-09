#if !defined(STOR_PARAMETER_H)
#define STOR_PARAMETER_H

// Created by Markus Klute on 2007 Jan 29.
// $Id: Parameter.h,v 1.5 2008/03/10 14:50:07 biery Exp $
//
// holds configuration parameter for StorageManager
//

#include <toolbox/net/Utils.h>
#include <string>

namespace stor 
{
  class Parameter
    {
    public:
      Parameter() :
	closeFileScript_("$CMSSW_RELEASE_BASE/src/EventFilter/StorageManager/scripts/perl/closeFile.pl"),
	notifyTier0Script_("$CMSSW_RELEASE_BASE/src/EventFilter/StorageManager/scripts/perl/notifyTier0.pl"),
	insertFileScript_("$CMSSW_RELEASE_BASE/src/EventFilter/StorageManager/scripts/perl/insertFile.pl"),
	fileCatalog_("summaryCatalog.txt"),
	smInstance_("0"),
	hostName_(toolbox::net::getHostName()),
	initialSafetyLevel_(0),
        fileName_("storageManager"),
        filePath_("/scratch2/cheung"),
        mailboxPath_("/scratch2/cheung/mbox"),
        setupLabel_("mtcc"),
	maxFileSize_(-1),
        highWaterMark_(0.9),
        lumiSectionTimeOut_(10.0),
        exactFileSizeTest_(false)
      {
        // strip domainame
         std::string::size_type pos = hostName_.find('.');  
         if ( pos != std::string::npos ) {  
           std::string basename = hostName_.substr(0,pos);  
           hostName_ = basename;
         }
      }

      const std::string& closeFileScript()    const {return closeFileScript_;}
      const std::string& notifyTier0Script()  const {return notifyTier0Script_;}
      const std::string& insertFileScript()   const {return insertFileScript_;}
      const std::string& fileCatalog()        const {return fileCatalog_;}
      const std::string& smInstance()         const {return smInstance_;}
      const std::string& host()               const {return hostName_;}
      const std::string& fileName()           const {return fileName_;}
      const std::string& filePath()           const {return filePath_;}
      const std::string& mailboxPath()        const {return mailboxPath_;}
      const std::string& setupLabel()         const {return setupLabel_;}
      int    maxFileSize()             const {return maxFileSize_;} 
      double highWaterMark()           const {return highWaterMark_;}
      double lumiSectionTimeOut()      const {return lumiSectionTimeOut_;}
      bool exactFileSizeTest()         const {return exactFileSizeTest_;}
      int initialSafetyLevel()         const {return initialSafetyLevel_;}

      // not efficient to pass object but tolerable
      void setCloseFileScript   (std::string x) {closeFileScript_=x;}
      void setNotifyTier0Script (std::string x) {notifyTier0Script_=x;}
      void setInsertFileScript  (std::string x) {insertFileScript_=x;}
      void setFileCatalog       (std::string x) {fileCatalog_=x;}
      void setSmInstance        (std::string x) {smInstance_=x;}
      void setfileName          (std::string x) {fileName_=x;}
      void setfilePath          (std::string x) {filePath_=x;}
      void setmailboxPath       (std::string x) {mailboxPath_=x;}
      void setsetupLabel        (std::string x) {setupLabel_=x;}
      void setmaxFileSize               (int x) {maxFileSize_=x;}
      void sethighWaterMark          (double x) {highWaterMark_=x;}
      void setlumiSectionTimeOut     (double x) {lumiSectionTimeOut_=x;}
      void setExactFileSizeTest        (bool x) {exactFileSizeTest_=x;}
      void initialSafetyLevel   (int i)         {initialSafetyLevel_=i;}

   private:
      std::string closeFileScript_;
      std::string notifyTier0Script_;
      std::string insertFileScript_;
      std::string fileCatalog_;
      std::string smInstance_;
      std::string hostName_;
      int         initialSafetyLevel_;
      std::string fileName_;
      std::string filePath_;
      std::string mailboxPath_;
      std::string setupLabel_;
      int         maxFileSize_; 
      double      highWaterMark_;
      double      lumiSectionTimeOut_;
      bool        exactFileSizeTest_;
    }; 
}

#endif

