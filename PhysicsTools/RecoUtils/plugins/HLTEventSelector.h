#ifndef Workspace_HLTEventSelector_h_
#define Workspace_HLTEventSelector_h_

/** Trivial example for a HLT selector.
 *  To be modified for analysis!
 */
// Original author: W. Adam, 10/4/08

// system include files
#include <memory>

// user include files
#include "PhysicsTools/UtilAlgos/interface/EventSelector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <vector>
#include <string>

class HLTEventSelector : public EventSelector {
public:
  HLTEventSelector (const edm::ParameterSet&);
  virtual bool select (const edm::Event&) const;
  virtual ~HLTEventSelector () {}
private:
  edm::InputTag triggerResults_;        ///< tag for input collection
  std::vector<std::string> pathNames_;  ///< trigger path names (ORed)
};
#endif
