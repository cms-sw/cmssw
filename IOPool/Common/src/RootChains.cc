/*----------------------------------------------------------------------
$Id: RootChains.cc,v 1.60 2007/09/07 19:34:31 wmtan Exp $
----------------------------------------------------------------------*/
#include "IOPool/Common/interface/RootChains.h"
#include "TChain.h"

#include "DataFormats/Provenance/interface/BranchType.h"

namespace edm {
  RootChains &
  RootChains::instance() {
    static RootChains chains;
    return chains;
  }
 
  void
  RootChains::makeChains() {
    if (!event_) event_ = boost::shared_ptr<TChain>(new TChain(BranchTypeToProductTreeName(InEvent).c_str()));
    if (!eventMeta_) eventMeta_ = boost::shared_ptr<TChain>(new TChain(BranchTypeToMetaDataTreeName(InEvent).c_str()));
    // if (!lumi_) lumi_ = boost::shared_ptr<TChain>(new TChain(BranchTypeToProductTreeName(InLumi).c_str()));
    // if (!lumiMeta_) lumiMeta_ = boost::shared_ptr<TChain>(new TChain(BranchTypeToMetaDataTreeName(InLumi).c_str()));
    // if (!run_) run_ = boost::shared_ptr<TChain>(new TChain(BranchTypeToProductTreeName(InRun).c_str()));
    // if (!runMeta_) runMeta_ = boost::shared_ptr<TChain>(new TChain(BranchTypeToMetaDataTreeName(InRun).c_str()));
  }

  void
  RootChains::addFile(std::string const& fileName) {
    char const* fn = fileName.c_str();
    if (event_) event_->AddFile(fn);
    if (eventMeta_) eventMeta_->AddFile(fn);
    if (lumi_) lumi_->AddFile(fn);
    if (lumiMeta_) lumiMeta_->AddFile(fn);
    if (run_) run_->AddFile(fn);
    if (runMeta_) runMeta_->AddFile(fn);
  }
}
