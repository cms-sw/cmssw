#ifndef FWCore_Framework_PrincipalCache_h
#define FWCore_Framework_PrincipalCache_h

/*
Designed to save RunPrincipal's and LuminosityBlockPrincipal's
in memory.  Manages merging of products in those principals
when there is more than one principal from the same run
or luminosity block.

Also contains a non-owning pointer the EventPrincipal, which is reused each event.

Original Author: W. David Dagenhart
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"

#include <map>

namespace edm {

  class RunKey {
  public:
    int processHistoryIDIndex() const { return processHistoryIDIndex_; }
    int run() const { return run_; }

    RunKey(int index, int run) : processHistoryIDIndex_(index), run_(run) { }

    bool operator<(RunKey const& right) const {
      if (processHistoryIDIndex_ == right.processHistoryIDIndex_) {
        return run_ < right.run_;
      }
      return processHistoryIDIndex_ < right.processHistoryIDIndex_;
    }

  private:
    int processHistoryIDIndex_;
    int run_;
  };

  class LumiKey {
  public:
    int processHistoryIDIndex() const { return processHistoryIDIndex_; }
    int run() const { return run_; }
    int lumi() const { return lumi_; }

    LumiKey(int index, int run, int lumi) : processHistoryIDIndex_(index), run_(run), lumi_(lumi) { }

    bool operator<(const LumiKey& right) const {
      if (processHistoryIDIndex_ == right.processHistoryIDIndex_) {
        if (run_ == right.run_) return lumi_ < right.lumi_;
        return run_ < right.run_;
      }
      return processHistoryIDIndex_ < right.processHistoryIDIndex_;
    }

  private:
    int processHistoryIDIndex_;
    int run_;
    int lumi_;
  };

  class PrincipalCache {
  public:

    PrincipalCache();
    ~PrincipalCache();

    RunPrincipal& runPrincipal(ProcessHistoryID const& phid, int run);
    RunPrincipal const& runPrincipal(ProcessHistoryID const& phid, int run) const;
    boost::shared_ptr<RunPrincipal> runPrincipalPtr(ProcessHistoryID const& phid, int run);

    // Current run (most recently read and inserted run)
    RunPrincipal& runPrincipal();
    RunPrincipal const& runPrincipal() const;
    boost::shared_ptr<RunPrincipal> runPrincipalPtr();

    LuminosityBlockPrincipal& lumiPrincipal(ProcessHistoryID const& phid, int run, int lumi);
    LuminosityBlockPrincipal const& lumiPrincipal(ProcessHistoryID const& phid, int run, int lumi) const;
    boost::shared_ptr<LuminosityBlockPrincipal> lumiPrincipalPtr(ProcessHistoryID const& phid, int run, int lumi);

    // Current luminosity block (most recently read and inserted luminosity block)
    LuminosityBlockPrincipal& lumiPrincipal();
    LuminosityBlockPrincipal const& lumiPrincipal() const;
    boost::shared_ptr<LuminosityBlockPrincipal> lumiPrincipalPtr();

    // Event
    EventPrincipal& eventPrincipal() {return *eventPrincipal_;}
    EventPrincipal const& eventPrincipal() const {return *eventPrincipal_;}

    bool merge(boost::shared_ptr<RunAuxiliary> aux, boost::shared_ptr<ProductRegistry const> reg);
    bool merge(boost::shared_ptr<LuminosityBlockAuxiliary> aux, boost::shared_ptr<ProductRegistry const> reg);

    bool insert(boost::shared_ptr<RunPrincipal> rp);
    bool insert(boost::shared_ptr<LuminosityBlockPrincipal> lbp);
    void insert(boost::shared_ptr<EventPrincipal> ep) {eventPrincipal_ = ep;}

    bool noMoreRuns();
    bool noMoreLumis();

    RunPrincipal const& lowestRun() const;
    LuminosityBlockPrincipal const& lowestLumi() const;

    void deleteLowestRun();
    void deleteLowestLumi();

    void deleteRun(ProcessHistoryID const& phid, int run);
    void deleteLumi(ProcessHistoryID const& phid, int run, int lumi);

    void adjustEventToNewProductRegistry(boost::shared_ptr<ProductRegistry const> reg);

    void adjustIndexesAfterProductRegistryAddition();

  private:

    std::vector<ProcessHistoryID> processHistoryIDs_;
    std::map<ProcessHistoryID, int> processHistoryIDsMap_;

    typedef std::map<RunKey, boost::shared_ptr<RunPrincipal> >::iterator RunIterator;
    typedef std::map<RunKey, boost::shared_ptr<RunPrincipal> >::const_iterator ConstRunIterator;
    typedef std::map<LumiKey, boost::shared_ptr<LuminosityBlockPrincipal> >::iterator LumiIterator;
    typedef std::map<LumiKey, boost::shared_ptr<LuminosityBlockPrincipal> >::const_iterator ConstLumiIterator;

    std::map<RunKey, boost::shared_ptr<RunPrincipal> > runPrincipals_;
    std::map<LumiKey, boost::shared_ptr<LuminosityBlockPrincipal> > lumiPrincipals_;

    boost::shared_ptr<EventPrincipal> eventPrincipal_;
    boost::shared_ptr<RunPrincipal> currentRunPrincipal_;
    boost::shared_ptr<LuminosityBlockPrincipal> currentLumiPrincipal_;
  };
}

#endif
