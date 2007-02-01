#if !defined(STOR_PARAMETER_H)
#define STOR_PARAMETER_H

// Created by Markus Klute on 2007 Jan 29.
// $Id:$

// holds configuration parameter for StorageManager

// should be moved to EventFilter/StorageManager

#include <string>

namespace stor 
{
  class Parameter
    {
    public:
      Parameter():
	updateStatusScript_("$CMSSW_BASE/src/EventFilter/StorageManager/scripts/perl/updateStatus.pl"),
	notifyTier0Script_("$CMSSW_BASE/src/EventFilter/StorageManager/scripts/perl/notifyTier0.pl"),
	insertFileScript_("$CMSSW_BASE/src/EventFilter/StorageManager/scripts/perl/insertFile.pl"),
	fileCatalog_("summaryCatalog.txt"),
	smInstance_("0")
	{}

      std::string updateStatusScript() {return updateStatusScript_;}
      std::string notifyTier0Script()  {return notifyTier0Script_;}
      std::string insertFileScript()   {return insertFileScript_;}
      std::string fileCatalog()        {return fileCatalog_;}
      std::string smInstance()         {return smInstance_;}

      void setUpdateStatusScript(std::string x) {updateStatusScript_=x;}
      void setNotifyTier0Script (std::string x) {notifyTier0Script_=x;}
      void setInsertFileScript  (std::string x) {insertFileScript_=x;}
      void setFileCatalog       (std::string x) {fileCatalog_=x;}
      void setSmInstance        (std::string x) {smInstance_=x;}

    private:
      std::string updateStatusScript_;
      std::string notifyTier0Script_;
      std::string insertFileScript_;
      std::string fileCatalog_;
      std::string smInstance_;
    }; 
}

#endif

