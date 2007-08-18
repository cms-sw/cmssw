#if !defined(STOR_PARAMETER_H)
#define STOR_PARAMETER_H

// Created by Markus Klute on 2007 Jan 29.
// $Id: Parameter.h,v 1.2 2007/02/05 16:39:40 klute Exp $

// holds configuration parameter for StorageManager

// should be moved to EventFilter/StorageManager

#include <toolbox/net/Utils.h>
#include <string>

namespace stor 
{
  class Parameter
    {
    public:
      Parameter():
	closeFileScript_("$CMSSW_RELEASE_BASE/src/EventFilter/StorageManager/scripts/perl/closeFile.pl"),
	notifyTier0Script_("$CMSSW_RELEASE_BASE/src/EventFilter/StorageManager/scripts/perl/notifyTier0.pl"),
	insertFileScript_("$CMSSW_RELEASE_BASE/src/EventFilter/StorageManager/scripts/perl/insertFile.pl"),
	fileCatalog_("summaryCatalog.txt"),
	smInstance_("0"),
	initialSafetyLevel_(0),
        fileName_("storageManager"),
        filePath_("/scratch2/cheung"),
        mailboxPath_("/scratch2/cheung/mbox"),
        setupLabel_("mtcc"),
        highWaterMark_(0.9),
        lumiSectionTimeOut_(10.0)
	{
	  hostName_ = toolbox::net::getHostName();
	}

      std::string closeFileScript()    {return closeFileScript_;}
      std::string notifyTier0Script()  {return notifyTier0Script_;}
      std::string insertFileScript()   {return insertFileScript_;}
      std::string fileCatalog()        {return fileCatalog_;}
      std::string smInstance()         {return smInstance_;}
      std::string host()               {return hostName_;}
      std::string fileName()           {return fileName_;}
      std::string filePath()           {return filePath_;}
      std::string mailboxPath()        {return mailboxPath_;}
      std::string setupLabel()         {return setupLabel_;}
      double highWaterMark()           {return highWaterMark_;}
      double lumiSectionTimeOut()      {return lumiSectionTimeOut_;}

      int initialSafetyLevel()         {return initialSafetyLevel_;}

      void setCloseFileScript   (std::string x) {closeFileScript_=x;}
      void setNotifyTier0Script (std::string x) {notifyTier0Script_=x;}
      void setInsertFileScript  (std::string x) {insertFileScript_=x;}
      void setFileCatalog       (std::string x) {fileCatalog_=x;}
      void setSmInstance        (std::string x) {smInstance_=x;}
      void setfileName          (std::string x) {fileName_=x;}
      void setfilePath          (std::string x) {filePath_=x;}
      void setmailboxPath       (std::string x) {mailboxPath_=x;}
      void setsetupLabel        (std::string x) {setupLabel_=x;}
      void sethighWaterMark          (double x) {highWaterMark_=x;}
      void setlumiSectionTimeOut     (double x) {lumiSectionTimeOut_=x;}

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
      double highWaterMark_;
      double lumiSectionTimeOut_;
    }; 
}

#endif

