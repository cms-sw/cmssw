#ifndef Input_PoolSecondarySource_h
#define Input_PoolSecondarySource_h

/*----------------------------------------------------------------------

PoolSecondarySource: This is a SecondaryInputSource

$Id: PoolSecondarySource.h,v 1.7 2005/12/01 22:36:08 wmtan Exp $

----------------------------------------------------------------------*/

#include <map>
#include <vector>
#include <string>

#include "IOPool/Common/interface/PoolCatalog.h"
#include "IOPool/CommonInput/interface/RootFile.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/SecondaryInputSource.h"

#include "boost/shared_ptr.hpp"

namespace edm {

  class PoolSecondarySource : public SecondaryInputSource {
  public:
    explicit PoolSecondarySource(ParameterSet const& pset);
    virtual ~PoolSecondarySource();

  private:
    PoolSecondarySource(PoolSecondarySource const&); // disable copy construction
    PoolSecondarySource & operator=(PoolSecondarySource const&); // disable assignment
    virtual void read_(int idx, int number, std::vector<EventPrincipal*>& result);
    void init(std::string const& file);
    bool next();

    PoolCatalog catalog_;
    std::map<ProductID, BranchDescription> productMap_;
    std::vector<std::string> const files_;
    std::vector<std::string>::const_iterator fileIter_;
    boost::shared_ptr<RootFile> rootFile_;
  }; // class PoolSecondarySource

}
#endif
