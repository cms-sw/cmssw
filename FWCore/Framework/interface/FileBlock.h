#ifndef FWCore_Framework_FileBlock_h
#define FWCore_Framework_FileBlock_h

/*----------------------------------------------------------------------

FileBlock: Properties of an input file.

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "DataFormats/Provenance/interface/BranchChildren.h"
class TTree;
#include "boost/shared_ptr.hpp"
#include <map>
#include <string>
#include <vector>

namespace edm {
  class BranchDescription;
  class FileBlock {
  public:
    FileBlock() : 
      fileFormatVersion_(),
      tree_(0), metaTree_(0),
      lumiTree_(0), lumiMetaTree_(0),
      runTree_(0), runMetaTree_(0),
      fastCopyable_(false), fileName_(),
      branchChildren_(new BranchChildren) {}

    FileBlock(FileFormatVersion const& version,
	      TTree const* ev, TTree const* meta,
	      TTree const* lumi, TTree const* lumiMeta,
	      TTree const* run, TTree const* runMeta,
	      bool fastCopy,
	      std::string const& fileName,
	      boost::shared_ptr<BranchChildren> branchChildren) :
      fileFormatVersion_(version),
      tree_(const_cast<TTree *>(ev)), 
      metaTree_(const_cast<TTree *>(meta)), 
      lumiTree_(const_cast<TTree *>(lumi)), 
      lumiMetaTree_(const_cast<TTree *>(lumiMeta)), 
      runTree_(const_cast<TTree *>(run)), 
      runMetaTree_(const_cast<TTree *>(runMeta)), 
      fastCopyable_(fastCopy), 
      fileName_(fileName), 
      branchChildren_(branchChildren) {}
    
    ~FileBlock() {}

    FileFormatVersion const& fileFormatVersion() const {return fileFormatVersion_;}
    TTree * const tree() const {return tree_;}
    TTree * const metaTree() const {return metaTree_;}
    TTree * const lumiTree() const {return lumiTree_;}
    TTree * const lumiMetaTree() const {return lumiMetaTree_;}
    TTree * const runTree() const {return runTree_;}
    TTree * const runMetaTree() const {return runMetaTree_;}

    bool fastClonable() const {return fastCopyable_;}
    std::string const& fileName() const {return fileName_;}

    void setNotFastCopyable() {fastCopyable_ = false;}
    BranchChildren const& branchChildren() const { return *branchChildren_; }
    void close () {runMetaTree_ = lumiMetaTree_ = metaTree_ = runTree_ = lumiTree_ = tree_ = 0;}

  private:
    FileFormatVersion fileFormatVersion_;
    // We use bare pointers because ROOT owns these.
    TTree * tree_;
    TTree * metaTree_;
    TTree * lumiTree_;
    TTree * lumiMetaTree_;
    TTree * runTree_;
    TTree * runMetaTree_;
    bool fastCopyable_;
    std::string fileName_;
    boost::shared_ptr<BranchChildren> branchChildren_;
  };
}
#endif
