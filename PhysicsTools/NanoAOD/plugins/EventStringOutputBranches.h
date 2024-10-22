#ifndef PhysicsTools_NanoAOD_EventStringOutputBranches_h
#define PhysicsTools_NanoAOD_EventStringOutputBranches_h

#include <string>
#include <vector>
#include <TTree.h>
#include "FWCore/Framework/interface/EventForOutput.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

class EventStringOutputBranches {
public:
  EventStringOutputBranches(const edm::BranchDescription *desc,
                            const edm::EDGetToken &token,
                            bool update_only_at_new_lumi = false)
      : m_token(token), m_lastLumi(-1), m_fills(0), m_update_only_at_new_lumi(update_only_at_new_lumi) {
    if (desc->className() != "std::basic_string<char,std::char_traits<char> >")
      throw cms::Exception("Configuration",
                           "NanoAODOutputModule/EventStringOutputBranches can only write out std::string objects");
  }

  void updateEventStringNames(TTree &, const std::string &);
  void fill(const edm::EventForOutput &iEvent, TTree &tree);

private:
  edm::EDGetToken m_token;
  struct NamedBranchPtr {
    std::string name, title;
    TBranch *branch;
    bool buffer;
    NamedBranchPtr(const std::string &aname, const std::string &atitle, TBranch *branchptr = nullptr)
        : name(aname), title(atitle), branch(branchptr), buffer(false) {}
  };
  std::vector<NamedBranchPtr> m_evStringBranches;
  long m_lastLumi;
  unsigned long m_fills;
  bool m_update_only_at_new_lumi;
};

#endif
