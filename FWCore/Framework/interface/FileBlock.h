#ifndef FWCore_Framework_FileBlock_h
#define FWCore_Framework_FileBlock_h

/*----------------------------------------------------------------------

FileBlock: Properties of an input file.

$Id: FileBlock.h,v 1.7 2008/05/28 18:52:01 wdd Exp $

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
    FileBlock() : fileFormatVersion_(),
	tree_(0), metaTree_(0),
	lumiTree_(0), lumiMetaTree_(0),
	runTree_(0), runMetaTree_(0),
        fastCopyable_(false), fileName_(),
        sortedNewBranchNames_(), oldBranchNames_(),
	branchChildren_() {}
    FileBlock(FileFormatVersion const& version,
	TTree const* ev, TTree const* meta,
	TTree const* lumi, TTree const* lumiMeta,
	TTree const* run, TTree const* runMeta,
	bool fastCopy,
	std::string const& fileName,
	std::vector<std::string> const& newNames,
	std::vector<std::string> const& oldNames,
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
	  sortedNewBranchNames_(newNames),
	  oldBranchNames_(oldNames),
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
    std::vector<std::string> const& sortedNewBranchNames() const {return sortedNewBranchNames_;}
    std::vector<std::string> const& oldBranchNames() const {return oldBranchNames_;}

    void setNotFastCopyable() {fastCopyable_ = false;}

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
    std::vector<std::string> sortedNewBranchNames_;
    std::vector<std::string> oldBranchNames_;
    boost::shared_ptr<BranchChildren> branchChildren_;
  };
}
#endif
