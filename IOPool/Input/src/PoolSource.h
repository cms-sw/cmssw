#ifndef Input_PoolSource_h
#define Input_PoolSource_h

/*----------------------------------------------------------------------

PoolSource: This is an InputSource

$Id: PoolSource.h,v 1.27 2006/10/24 20:29:02 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include <vector>
#include <string>
#include <map>

#include "IOPool/Input/src/Inputfwd.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/VectorInputSource.h"
#include "DataFormats/Common/interface/BranchDescription.h"

#include "boost/shared_ptr.hpp"

namespace edm {

  class RootFile;
  class FileCatalogItem;
  class PoolSource : public VectorInputSource {
  public:
    explicit PoolSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~PoolSource();

    /// Called by framework at end of job
    virtual void endJob();

  private:
    typedef boost::shared_ptr<RootFile> RootFileSharedPtr;
    typedef input::EntryNumber EntryNumber;
    PoolSource(PoolSource const&); // disable copy construction
    PoolSource & operator=(PoolSource const&); // disable assignment
    virtual std::auto_ptr<EventPrincipal> read();
    virtual std::auto_ptr<EventPrincipal> readIt(EventID const& id);
    virtual void skip(int offset);
    virtual void rewind_();
    virtual void readMany_(int number, EventPrincipalVector& result);
    void init(FileCatalogItem const& file);
    void updateProductRegistry() const;
    void setInitialPosition(ParameterSet const& pset);
    bool next();
    bool previous();

    std::vector<FileCatalogItem>::const_iterator fileIter_;
    RootFileSharedPtr rootFile_;
    RootFileSharedPtr origRootFile_;
    BranchDescription::MatchMode matchMode_;
  }; // class PoolSource
  typedef PoolSource PoolRASource;
}
#endif
