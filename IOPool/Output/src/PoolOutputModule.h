#ifndef Output_PoolOutputModule_h
#define Output_PoolOutputModule_h

//////////////////////////////////////////////////////////////////////
//
// $Id: PoolOutputModule.h,v 1.15 2006/12/01 03:38:17 wmtan Exp $
//
// Class PoolOutputModule. Output module to POOL file
//
// Oringinal Author: Luca Lista
// Current Author: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include <memory>
#include <string>
#include <iosfwd>
#include "boost/array.hpp"

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
    explicit PoolOutputModule(ParameterSet const& ps);
    virtual ~PoolOutputModule();
    std::string const& fileName() const {return catalog_.fileName();}
    std::string const& logicalFileName() const {return catalog_.logicalFileName();}

  private:
    pool::IDataSvc *context() const {return context_.context();}

  private:
    virtual void beginJob(EventSetup const&);
    virtual void endJob();
    virtual void write(EventPrincipal const& e);
    virtual void endLuminosityBlock(LuminosityBlockPrincipal const& lb);
    virtual void endRun(RunPrincipal const& r);

    mutable OutputFileCatalog catalog_;
    mutable PoolDataSvc context_;
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
    void writeLuminosityBlock(LuminosityBlockPrincipal const& lb);
    void writeRun(RunPrincipal const& r);

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
      OutputItem() : branchDescription_(0), selected_(false), provenancePlacement_(), productPlacement_() {}
      OutputItem(BranchDescription const* bd, bool sel, pool::Placement const& plProv,
		pool::Placement const& plEvent = pool::Placement()) :
		branchDescription_(bd), selected_(sel),
		provenancePlacement_(plProv), productPlacement_(plEvent) {}
      ~OutputItem() {}
      BranchDescription const* branchDescription_;
      bool selected_;
      pool::Placement provenancePlacement_;
      pool::Placement productPlacement_;
    };
    typedef std::vector<OutputItem> OutputItemList;
    typedef boost::array<OutputItemList, EndBranchType> OutputItemListArray;
    typedef boost::array<std::vector<std::string>, EndBranchType> BranchNamesArray;

    void fillBranches(OutputItemList const& items, DataBlockImpl const& dataBlock) const;

    OutputItemListArray outputItemList_;
    BranchNamesArray branchNames_;
    std::string file_;
    std::string lfn_;
    JobReport::Token reportToken_;
    unsigned long eventCount_;
    unsigned long fileSizeCheckEvent_;
    boost::array<pool::Placement, EndBranchType> auxiliaryPlacement_;
    pool::Placement productDescriptionPlacement_;
    pool::Placement parameterSetPlacement_;
    pool::Placement moduleDescriptionPlacement_;
    pool::Placement processHistoryPlacement_;
    pool::Placement fileFormatVersionPlacement_;
    PoolOutputModule const* om_;
    mutable std::list<BranchEntryDescription> provenances_;
  };
}

#endif
