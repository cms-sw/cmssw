#ifndef Output_PoolOutputModule_h
#define Output_PoolOutputModule_h

//////////////////////////////////////////////////////////////////////
//
// $Id: PoolOutputModule.h,v 1.28 2007/08/28 18:22:39 wmtan Exp $
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

namespace edm {
  class ParameterSet;

  class PoolOutputModule : public OutputModule {
  public:
    friend class RootOutputFile;
    explicit PoolOutputModule(ParameterSet const& ps);
    virtual ~PoolOutputModule();
    std::string const& fileName() const {return fileName_;}
    std::string const& logicalFileName() const {return logicalFileName_;}
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

    std::string const fileName_;
    std::string const logicalFileName_;
    std::string const catalog_;
    unsigned int const maxFileSize_;
    int const compressionLevel_;
    int const basketSize_;
    int const splitLevel_;
    std::string const moduleLabel_;
    int fileCount_;
    boost::shared_ptr<RootOutputFile> rootFile_;
  };
}

#endif
