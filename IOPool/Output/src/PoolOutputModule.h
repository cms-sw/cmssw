#ifndef Output_PoolOutputModule_h
#define Output_PoolOutputModule_h

//////////////////////////////////////////////////////////////////////
//
// $Id: PoolOutputModule.h,v 1.10 2006/08/07 22:10:14 wmtan Exp $
//
// Class PoolOutputModule. Output module to POOL file
//
// Author: Luca Lista
// Co-Author: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include <memory>
#include <string>
#include <iosfwd>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Framework/interface/FileCatalog.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "IOPool/Common/interface/PoolDataSvc.h"
#include "FWCore/Utilities/interface/PersistentNames.h"
#include "PersistencySvc/Placement.h"

namespace pool {
  class IDataSvc;
}

namespace edm {
  class OutputFileCatalog;
  class ParameterSet;

  class PoolOutputModule : public OutputModule {
  public:
    class PoolFile;
    friend class PoolOutputModule::PoolFile;
    explicit PoolOutputModule(ParameterSet const& ps);
    virtual ~PoolOutputModule();
    virtual void beginJob(EventSetup const&);
    virtual void endJob();
    virtual void write(EventPrincipal const& e);

  private:
    pool::IDataSvc *context() const {return context_.context();}

  private:
    mutable OutputFileCatalog catalog_;
    mutable PoolDataSvc context_;
    std::string const fileName_;
    std::string const logicalFileName_;
    unsigned long commitInterval_;
    unsigned long maxFileSize_;
    int compressionLevel_;
    std::string const moduleLabel_;
    unsigned long fileCount_;
    boost::shared_ptr<PoolFile> poolFile_;
  };

  class PoolOutputModule::PoolFile {
  public:
    explicit PoolFile(PoolOutputModule * om);
    ~PoolFile() {}
    bool writeOne(EventPrincipal const& e);
    void endFile();

  private:
    void makePlacement(std::string const& treeName, std::string const& branchName,
       pool::Placement& placement);
    pool::IDataSvc *context() const {return om_->context();}
    void startTransaction() const;
    void commitTransaction() const;
    void commitAndFlushTransaction() const;
    void setBranchAliases() const;

  private:
    struct OutputItem {
      OutputItem() : branchDescription_(0), selected_(false), provenancePlacement_(), eventPlacement_() {}
      OutputItem(BranchDescription const* bd, bool sel, pool::Placement const& plProv,
		pool::Placement const& plEvent = pool::Placement()) :
		branchDescription_(bd), selected_(sel),
		provenancePlacement_(plProv), eventPlacement_(plEvent) {}
      ~OutputItem() {}
      BranchDescription const* branchDescription_;
      bool selected_;
      pool::Placement provenancePlacement_;
      pool::Placement eventPlacement_;
    };
    typedef std::vector<OutputItem> OutputItemList;
    OutputItemList outputItemList_;
    std::vector<std::string> branchNames_;
    std::string file_;
    std::string lfn_;
    JobReport::Token reportToken_;
    unsigned long eventCount_;
    unsigned long fileSizeCheckEvent_;
    pool::Placement auxiliaryPlacement_;
    pool::Placement productDescriptionPlacement_;
    pool::Placement parameterSetPlacement_;
    pool::Placement moduleDescriptionPlacement_;
    pool::Placement processHistoryPlacement_;
    pool::Placement fileFormatVersionPlacement_;
    pool::Placement runBlockPlacement_;
    pool::Placement luminosityBlockPlacement_;
    PoolOutputModule const* om_;
  };
}

#endif
