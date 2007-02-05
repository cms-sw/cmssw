#if !defined(STOR_PARAMETER_H)
#define STOR_PARAMETER_H

// Created by Markus Klute on 2007 Jan 29.
// $Id: Parameter.h,v 1.1 2007/02/05 11:19:56 klute Exp $

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
	initialSafetyLevel_(0)
	{
	  hostName_ = toolbox::net::getHostName();
	}

      std::string closeFileScript()    {return closeFileScript_;}
      std::string notifyTier0Script()  {return notifyTier0Script_;}
      std::string insertFileScript()   {return insertFileScript_;}
      std::string fileCatalog()        {return fileCatalog_;}
      std::string smInstance()         {return smInstance_;}
      std::string host()               {return hostName_;}

      int initialSafetyLevel()         {return initialSafetyLevel_;}

      void setCloseFileScript   (std::string x) {closeFileScript_=x;}
      void setNotifyTier0Script (std::string x) {notifyTier0Script_=x;}
      void setInsertFileScript  (std::string x) {insertFileScript_=x;}
      void setFileCatalog       (std::string x) {fileCatalog_=x;}
      void setSmInstance        (std::string x) {smInstance_=x;}

      void initialSafetyLevel   (int i)         {initialSafetyLevel_=i;}

   private:
      std::string closeFileScript_;
      std::string notifyTier0Script_;
      std::string insertFileScript_;
      std::string fileCatalog_;
      std::string smInstance_;
      std::string hostName_;
      int         initialSafetyLevel_;
    }; 
}

#endif

