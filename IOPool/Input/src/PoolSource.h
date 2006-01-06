#ifndef Input_PoolSource_h
#define Input_PoolSource_h

/*----------------------------------------------------------------------

PoolSource: This is an InputSource

$Id: PoolSource.h,v 1.9 2005/12/01 22:35:23 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include <vector>
#include <string>

#include "IOPool/Common/interface/PoolCatalog.h"
#include "IOPool/CommonInput/interface/RootDelayedReader.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/RandomAccessInputSource.h"

#include "boost/shared_ptr.hpp"

namespace edm {

  class RootFile;
  class PoolRASource : public RandomAccessInputSource {
  public:
    explicit PoolRASource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~PoolRASource();

  private:
    PoolRASource(PoolRASource const&); // disable copy construction
    PoolRASource & operator=(PoolRASource const&); // disable assignment
    virtual std::auto_ptr<EventPrincipal> read();
    virtual std::auto_ptr<EventPrincipal> read(EventID const& id);
    virtual void skip(int offset);
    void init(std::string const& file);
    bool next();

    PoolCatalog catalog_;
    RootDelayedReader::ProductMap productMap_;
    std::string const file_;
    std::vector<std::string> const files_;
    std::vector<std::string>::const_iterator fileIter_;
    boost::shared_ptr<RootFile> rootFile_;
    RootDelayedReader::EntryNumber remainingEvents_;
  }; // class PoolRASource
}
#endif
