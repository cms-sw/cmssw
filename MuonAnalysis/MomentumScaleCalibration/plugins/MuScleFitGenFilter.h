// System include files
// --------------------
#include <memory>
#include <vector>
#include <string>

// User include files
// ------------------
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

// Class declaration
// -----------------

class MuScleFitGenFilter : public edm::EDFilter {
 public:
  explicit MuScleFitGenFilter(const edm::ParameterSet&);
  ~MuScleFitGenFilter();

 private:
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob() {};

  std::string genParticlesName_;
  unsigned int totalEvents_;
  unsigned int eventsPassingTheFilter_;
};
