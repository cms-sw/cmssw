#ifndef IOPool_Input_PoolSource_h
#define IOPool_Input_PoolSource_h

/*----------------------------------------------------------------------

PoolSource: This is an InputSource

$Id: PoolSource.h,v 1.38 2007/07/23 23:42:36 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include <vector>
#include <string>
#include <string>

#include "Inputfwd.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Sources/interface/VectorInputSource.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"

#include "boost/shared_ptr.hpp"

namespace CLHEP {
  class RandFlat;
}

class TChain;

namespace edm {

  class RootFile;
  class FileCatalogItem;
  class PoolSource : public VectorInputSource {
  public:
    explicit PoolSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~PoolSource();

    /// Called by framework at end of job
    virtual void endJob();

    class RootChains {
      friend class PoolSource;
      friend class PoolOutputModule;
    private:
      RootChains() : event_(), eventMeta_(), lumi_(), lumiMeta_(), run_(), runMeta_() {}
      ~RootChains() {}

      void addFile(std::string const& fileName);
      void makeChains();
      static RootChains & instance();

      boost::shared_ptr<TChain> event_;
      boost::shared_ptr<TChain> eventMeta_;
      boost::shared_ptr<TChain> lumi_;
      boost::shared_ptr<TChain> lumiMeta_;
      boost::shared_ptr<TChain> run_;
      boost::shared_ptr<TChain> runMeta_;
    };

  private:
    typedef boost::shared_ptr<RootFile> RootFileSharedPtr;
    typedef input::EntryNumber EntryNumber;
    PoolSource(PoolSource const&); // disable copy construction
    PoolSource & operator=(PoolSource const&); // disable assignment
    virtual std::auto_ptr<EventPrincipal> read();
    virtual std::auto_ptr<EventPrincipal> readEvent_(boost::shared_ptr<LuminosityBlockPrincipal> lbp);
    virtual boost::shared_ptr<LuminosityBlockPrincipal> readLuminosityBlock_(boost::shared_ptr<RunPrincipal> rp);
    virtual boost::shared_ptr<RunPrincipal> readRun_();
    virtual std::auto_ptr<EventPrincipal> readIt(EventID const& id);
    virtual void skip(int offset);
    virtual void rewind_();
    virtual void readMany_(int number, EventPrincipalVector& result);
    void init(FileCatalogItem const& file);
    void updateProductRegistry() const;
    void setInitialPosition(ParameterSet const& pset);
    bool nextFile();
    bool next();
    bool previous();
    void rewindFile();
    void readRandom(int number, EventPrincipalVector& result);
    void randomize();

    std::vector<FileCatalogItem>::const_iterator fileIter_;
    RootFileSharedPtr rootFile_;
    BranchDescription::MatchMode matchMode_;

    CLHEP::RandFlat * flatDistribution_;
    int eventsRemainingInFile_;
  }; // class PoolSource
  typedef PoolSource PoolRASource;
}
#endif
