#ifndef Output_PoolOutputModule_h
#define Output_PoolOutputModule_h

//////////////////////////////////////////////////////////////////////
//
// $Id: PoolOutputModule.h,v 1.15 2005/10/12 02:33:07 wmtan Exp $
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
#include "PersistencySvc/Placement.h"

namespace pool {
  class IDataSvc;
}

namespace edm {
  class PoolCatalog;
  class ParameterSet;

  class PoolOutputModule : public OutputModule {
  public:
    explicit PoolOutputModule(ParameterSet const& ps);
    virtual ~PoolOutputModule();
    virtual void beginJob(EventSetup const&);
    virtual void endJob();
    virtual void write(EventPrincipal const& e);

  private:
    void makePlacement(std::string const& treeName, std::string const& branchName,
       pool::Placement& placement);
    void startTransaction() const;
    void commitTransaction() const;
    void setBranchAliases() const;

  private:
    typedef std::pair<BranchDescription const*, pool::Placement> OutputItem;
    typedef std::vector<OutputItem> OutputItemList;
    PoolCatalog catalog_;
    pool::IDataSvc * context_;
    OutputItemList outputItemList_;
    std::string const file_;
    std::string const lfn_;
    pool::Placement provenancePlacement_;
    pool::Placement auxiliaryPlacement_;
    pool::Placement productDescriptionPlacement_;
  };
}

#endif
