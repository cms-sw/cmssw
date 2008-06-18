#ifndef IOPool_Input_PoolSource_h
#define IOPool_Input_PoolSource_h

/*----------------------------------------------------------------------

PoolSource: This is an InputSource

----------------------------------------------------------------------*/

#include <memory>
#include <vector>
#include <string>

#include "Inputfwd.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Sources/interface/VectorInputSource.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchID.h"

#include "boost/scoped_ptr.hpp"

namespace edm {

  class RootInputFileSequence;
  class FileCatalogItem;
  class PoolSource : public VectorInputSource {
  public:
    explicit PoolSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~PoolSource();
    using InputSource::productRegistryUpdate;
    using InputSource::runPrincipal;

  private:
    typedef boost::shared_ptr<RootFile> RootFileSharedPtr;
    typedef input::EntryNumber EntryNumber;
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
    virtual void dropUnwantedBranches_(std::vector<std::string> const& wantedBranches);

    boost::scoped_ptr<RootInputFileSequence> primaryFileSequence_;
    boost::scoped_ptr<RootInputFileSequence> secondaryFileSequence_;
    std::vector<BranchID> branchIDsToReplace_;
  }; // class PoolSource
  typedef PoolSource PoolRASource;
}
#endif
