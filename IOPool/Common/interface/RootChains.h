#ifndef IOPool_Common_RootChains_h
#define IOPool_COmmon_RootChains_h

/*----------------------------------------------------------------------

RootChains: Class to allow for fast cloning of trees

$Id: RootChains.h,v 1.39 2007/09/07 19:34:31 wmtan Exp $

----------------------------------------------------------------------*/

#include <string>

#include "boost/shared_ptr.hpp"

class TChain;

namespace edm {

  class RootChains {
    friend class PoolSource;
    friend class RootOutputFile;
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
}
#endif
