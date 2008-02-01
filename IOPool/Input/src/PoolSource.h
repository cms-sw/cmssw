#ifndef IOPool_Input_PoolSource_h
#define IOPool_Input_PoolSource_h

/*----------------------------------------------------------------------

PoolSource: This is an InputSource

$Id: PoolSource.h,v 1.49 2008/01/08 06:57:39 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include <vector>
#include <string>

#include "Inputfwd.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Sources/interface/VectorInputSource.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/RunID.h"

#include "boost/shared_ptr.hpp"

namespace CLHEP {
  class RandFlat;
}

namespace edm {

  class RootFile;
  class FileCatalogItem;
  class PoolSource : public VectorInputSource {
  public:
    explicit PoolSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~PoolSource();

  private:
    typedef boost::shared_ptr<RootFile> RootFileSharedPtr;
    typedef input::EntryNumber EntryNumber;
    PoolSource(PoolSource const&); // disable copy construction
    PoolSource & operator=(PoolSource const&); // disable assignment
    virtual std::auto_ptr<EventPrincipal> readCurrentEvent();
    virtual std::auto_ptr<EventPrincipal> readEvent_(boost::shared_ptr<LuminosityBlockPrincipal> lbp);
    virtual boost::shared_ptr<LuminosityBlockPrincipal> readLuminosityBlock_();
    virtual boost::shared_ptr<RunPrincipal> readRun_();
    virtual boost::shared_ptr<FileBlock> readFile_();
    virtual void closeFile_();
    virtual void endJob();
    virtual InputSource::ItemType getNextItemType();
    virtual std::auto_ptr<EventPrincipal> readIt(EventID const& id);
    virtual void skip(int offset);
    virtual void rewind_();
    virtual void readMany_(int number, EventPrincipalVector& result);
    virtual void readMany_(int number, EventPrincipalVector& result, EventID const& id, unsigned int fileSeqNumber);
    virtual void readManyRandom_(int number, EventPrincipalVector& result, unsigned int& fileSeqNumber);
    void initFile(bool skipBadFiles);
    void updateProductRegistry() const;
    bool nextFile();
    bool previousFile();
    void rewindFile();

    bool firstFile_;
    std::vector<FileCatalogItem>::const_iterator fileIterBegin_;
    std::vector<FileCatalogItem>::const_iterator fileIter_;
    RootFileSharedPtr rootFile_;
    BranchDescription::MatchMode matchMode_;

    CLHEP::RandFlat * flatDistribution_;
    int eventsRemainingInFile_;
    RunNumber_t startAtRun_;
    LuminosityBlockNumber_t startAtLumi_;
    EventNumber_t startAtEvent_;
    unsigned int eventsToSkip_;
    bool skipBadFiles_;
    int forcedRunOffset_;
    RunNumber_t setRun_;
  }; // class PoolSource
  typedef PoolSource PoolRASource;
}
#endif
