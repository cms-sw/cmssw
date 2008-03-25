#ifndef FWCore_Catalog_OutputFileCatalog_h
#define FWCore_Catalog_OutputFileCatalog_h
//////////////////////////////////////////////////////////////////////
//
// $Id: OutputFileCatalog.h,v 1.1 2007/03/04 04:43:30 wmtan Exp $
//
// Class OutputFileCatalog. Common services to manage OutputFile catalog
//
// Author of original version: Luca Lista
// Author of current version: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include <string>
#include "FWCore/Catalog/interface/FileCatalog.h"

namespace edm {
  class OutputFileCatalog : public FileCatalog {
  public:
    explicit OutputFileCatalog(ParameterSet const& pset);
    virtual ~OutputFileCatalog();
    std::string const& logicalFileName() const {return logicalFileName_;}
    std::string const& fileName() const {return fileName_;}
    pool::FileCatalog::FileID registerFile(std::string const& pfn, std::string const& lfn);
  private:
    std::string fileName_;
    std::string logicalFileName_;
  };
}

#endif
