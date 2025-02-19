// System include files
// --------------------
#include <memory>
#include <vector>

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

class MuScleFitFilter : public edm::EDFilter {
 public:
  explicit MuScleFitFilter(const edm::ParameterSet&);
  ~MuScleFitFilter();

 private:
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob() {};

  // Member data
  // -----------
  int eventsRead;
  int eventsWritten;
  bool debug;
  int theMuonType;
  std::vector<double> Mmin;
  std::vector<double> Mmax;
  int maxWrite;
  unsigned int minimumMuonsNumber;

  // Collections labels
  // ------------------
  edm::InputTag theMuonLabel;

};
