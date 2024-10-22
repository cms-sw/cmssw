#include "PhysicsTools/NanoAOD/plugins/EventStringOutputBranches.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"

#include <iostream>

void EventStringOutputBranches::updateEventStringNames(TTree &tree, const std::string &evstring) {
  bool found = false;
  for (auto &existing : m_evStringBranches) {
    existing.buffer = false;
    if (evstring == existing.name) {
      existing.buffer = true;
      found = true;
    }
  }
  if (!found && (!evstring.empty())) {
    NamedBranchPtr nb(evstring, "EventString bit");
    bool backFillValue = false;
    nb.branch = tree.Branch(nb.name.c_str(), &backFillValue, (nb.name + "/O").c_str());
    nb.branch->SetTitle(nb.title.c_str());
    for (size_t i = 0; i < m_fills; i++)
      nb.branch->Fill();  // Back fill
    nb.buffer = true;
    m_evStringBranches.push_back(nb);
    for (auto &existing : m_evStringBranches)
      existing.branch->SetAddress(&(existing.buffer));  // m_evStringBranches might have been resized
  }
}

void EventStringOutputBranches::fill(const edm::EventForOutput &iEvent, TTree &tree) {
  if ((!m_update_only_at_new_lumi) || m_lastLumi != iEvent.id().luminosityBlock()) {
    edm::Handle<std::string> handle;
    iEvent.getByToken(m_token, handle);
    const std::string &evstring = *handle;
    m_lastLumi = iEvent.id().luminosityBlock();
    updateEventStringNames(tree, evstring);
  }
  m_fills++;
}
