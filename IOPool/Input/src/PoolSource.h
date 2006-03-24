#ifndef Input_PoolSource_h
#define Input_PoolSource_h

/*----------------------------------------------------------------------

PoolSource: This is an InputSource

$Id: PoolSource.h,v 1.16 2006/03/14 23:33:01 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include <vector>
#include <string>
#include <map>

#include "IOPool/Common/interface/PoolCatalog.h"
#include "IOPool/Input/src/Inputfwd.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/VectorInputSource.h"

#include "boost/shared_ptr.hpp"

namespace edm {

  class RootFile;
  class PoolRASource : public VectorInputSource {
  public:
    explicit PoolRASource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~PoolRASource();

  private:
    typedef boost::shared_ptr<RootFile> RootFileSharedPtr;
    typedef std::map<std::string, RootFileSharedPtr> RootFileMap;
    typedef input::EntryNumber EntryNumber;
    PoolRASource(PoolRASource const&); // disable copy construction
    PoolRASource & operator=(PoolRASource const&); // disable assignment
    virtual std::auto_ptr<EventPrincipal> read();
    virtual std::auto_ptr<EventPrincipal> read(EventID const& id);
    virtual void skip(int offset);
    virtual void readMany_(int number, EventPrincipalVector& result);
    void init(std::string const& file);
    void updateProductRegistry() const;
    bool next();
    bool previous();

    PoolCatalog catalog_;
    std::vector<std::string> const files_;
    std::vector<std::string>::const_iterator fileIter_;
    RootFileSharedPtr rootFile_;
    RootFileMap rootFiles_;
    int maxEvents_;
    int remainingEvents_;
    bool mainInput_;
  }; // class PoolRASource
}
#endif
