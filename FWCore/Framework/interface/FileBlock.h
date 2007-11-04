#ifndef FWCore_Framework_FileBlock_h
#define FWCore_Framework_FileBlock_h

/*----------------------------------------------------------------------

FileBlock: Properties of an input file.

$Id: FileBlock.h,v 1.2 2007/11/03 06:52:54 wmtan Exp $

----------------------------------------------------------------------*/

class TTree;
#include <map>
#include <string>
#include <vector>

namespace edm {
  class BranchDescription;
  class FileBlock {
  public:
    FileBlock() : tree_(0), metaTree_(0), fastClonable_(false), sortedNewBranchNames_(), oldBranchNames_() {}
    FileBlock(TTree const* ev, TTree const* meta, bool fastClone,
	std::vector<std::string> const& newNames,
	std::vector<std::string> const& oldNames) :
	  tree_(const_cast<TTree *>(ev)), 
	  metaTree_(const_cast<TTree *>(meta)), 
	  fastClonable_(fastClone), 
	  sortedNewBranchNames_(newNames),
	  oldBranchNames_(oldNames) {}
    ~FileBlock() {}

    TTree * const tree() const {return tree_;}
    TTree * const metaTree() const {return metaTree_;}
    bool fastClonable() const {return fastClonable_;}
    void setNonClonable() {fastClonable_ = false;}
    std::vector<std::string> const& sortedNewBranchNames() const {return sortedNewBranchNames_;}
    std::vector<std::string> const& oldBranchNames() const {return oldBranchNames_;}

  private:
    // We use bare pointers because ROOT owns these.
    TTree * tree_;
    TTree * metaTree_;
    bool fastClonable_;
    std::vector<std::string> sortedNewBranchNames_;
    std::vector<std::string> oldBranchNames_;
  };
}
#endif
