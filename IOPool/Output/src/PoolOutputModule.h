#ifndef Output_PoolOutputModule_h
#define Output_PoolOutputModule_h

//////////////////////////////////////////////////////////////////////
//
// $Id: PoolOutputModule.h,v 1.27 2007/08/28 14:28:52 wmtan Exp $
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
#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Catalog/interface/OutputFileCatalog.h"

namespace edm {
  class ParameterSet;

  class PoolOutputModule : public OutputModule {
  public:
    friend class RootOutputFile;
    explicit PoolOutputModule(ParameterSet const& ps);
    virtual ~PoolOutputModule();
    std::string const& fileName() const {return catalog_.fileName();}
    std::string const& logicalFileName() const {return catalog_.logicalFileName();}
    int const& compressionLevel() const {return compressionLevel_;}
    int const& basketSize() const {return basketSize_;}
    int const& splitLevel() const {return splitLevel_;}

  private:
    virtual void beginJob(EventSetup const&);
    virtual void endJob();
    virtual void write(EventPrincipal const& e);
    virtual void endLuminosityBlock(LuminosityBlockPrincipal const& lb);
    virtual void beginRun(RunPrincipal const& r);
    virtual void endRun(RunPrincipal const& r);

    mutable OutputFileCatalog catalog_;
    unsigned int maxFileSize_;
    int compressionLevel_;
    int basketSize_;
    int splitLevel_;
    std::string const moduleLabel_;
    int fileCount_;
    boost::shared_ptr<RootOutputFile> rootFile_;
  };
}

#endif
