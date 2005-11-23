#ifndef Output_PoolOutputModule_h
#define Output_PoolOutputModule_h

//////////////////////////////////////////////////////////////////////
//
// $Id: PoolOutputModule.h,v 1.2 2005/11/21 19:57:42 wmtan Exp $
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
#include "IOPool/Common/interface/PoolNames.h"
#include "IOPool/Common/interface/PoolCatalog.h"
#include "IOPool/Common/interface/PoolDataSvc.h"
#include "PersistencySvc/Placement.h"

namespace pool {
  class IDataSvc;
}

namespace edm {
  class PoolCatalog;
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
    PoolCatalog mutable catalog_;
    PoolDataSvc mutable context_;
    std::string const fileName_;
    std::string const logicalFileName_;
    unsigned long commitInterval_;
    unsigned long maxFileSize_;
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
    typedef std::pair<BranchDescription const*, pool::Placement> OutputItem;
    typedef std::vector<OutputItem> OutputItemList;
    OutputItemList outputItemList_;
    std::string file_;
    std::string lfn_;
    unsigned long eventCount_;
    unsigned long fileSizeCheckEvent_;
    pool::Placement provenancePlacement_;
    pool::Placement auxiliaryPlacement_;
    pool::Placement productDescriptionPlacement_;
    PoolOutputModule const* om_;
  }; 
}

#endif
