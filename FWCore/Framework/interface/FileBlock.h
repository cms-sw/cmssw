#ifndef FWCore_Framework_FileBlock_h
#define FWCore_Framework_FileBlock_h

/*----------------------------------------------------------------------

FileBlock: Properties of an input file.

$Id: FileBlock.h,v 1.3 2007/11/04 02:45:07 wmtan Exp $

----------------------------------------------------------------------*/

class TTree;
#include <map>
#include <string>
#include <vector>

namespace edm {
  class BranchDescription;
  class FileBlock {
  public:
    FileBlock() : tree_(0), metaTree_(0),
         lumiTree_(0), lumiMetaTree_(0),
         runTree_(0), runMetaTree_(0),
	 fastClonable_(false), sortedNewBranchNames_(), oldBranchNames_() {}
    FileBlock(TTree const* ev, TTree const* meta,
	TTree const* lumi, TTree const* lumiMeta,
	TTree const* run, TTree const* runMeta,
	bool fastClone,
	std::vector<std::string> const& newNames,
	std::vector<std::string> const& oldNames) :
	  tree_(const_cast<TTree *>(ev)), 
	  metaTree_(const_cast<TTree *>(meta)), 
	  lumiTree_(const_cast<TTree *>(lumi)), 
	  lumiMetaTree_(const_cast<TTree *>(lumiMeta)), 
	  runTree_(const_cast<TTree *>(run)), 
	  runMetaTree_(const_cast<TTree *>(runMeta)), 
	  fastClonable_(fastClone), 
	  sortedNewBranchNames_(newNames),
	  oldBranchNames_(oldNames) {}
    ~FileBlock() {}

    TTree * const tree() const {return tree_;}
    TTree * const metaTree() const {return metaTree_;}
    TTree * const lumiTree() const {return lumiTree_;}
    TTree * const lumiMetaTree() const {return lumiMetaTree_;}
    TTree * const runTree() const {return runTree_;}
    TTree * const runMetaTree() const {return runMetaTree_;}
    bool fastClonable() const {return fastClonable_;}
    std::vector<std::string> const& sortedNewBranchNames() const {return sortedNewBranchNames_;}
    std::vector<std::string> const& oldBranchNames() const {return oldBranchNames_;}

  private:
    // We use bare pointers because ROOT owns these.
    TTree * tree_;
    TTree * metaTree_;
    TTree * lumiTree_;
    TTree * lumiMetaTree_;
    TTree * runTree_;
    TTree * runMetaTree_;
    bool fastClonable_;
    std::vector<std::string> sortedNewBranchNames_;
    std::vector<std::string> oldBranchNames_;
  };
}
#endif
