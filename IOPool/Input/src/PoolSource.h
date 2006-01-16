#ifndef Input_PoolSource_h
#define Input_PoolSource_h

/*----------------------------------------------------------------------

PoolSource: This is an InputSource

$Id: PoolSource.h,v 1.12 2006/01/07 20:42:33 wmtan Exp $

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
    virtual void readMany_(int number, std::vector<EventPrincipal *>& result);
    void init(std::string const& file);
    void updateRegistry() const;
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
