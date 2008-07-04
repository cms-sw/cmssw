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
#include "FWCore/ParameterSet/interface/InputTag.h"

// Class declaration
// -----------------

class MuScleFitFilter : public edm::EDFilter {
 public:
  explicit MuScleFitFilter(const edm::ParameterSet&);
  ~MuScleFitFilter();

 private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  // Member data
  // -----------
  int eventsRead;
  int eventsWritten;
  bool debug;
  int theMuonType;
  double Mmin;
  double Mmax;
  int maxWrite;

  // Collections labels
  // ------------------
  edm::InputTag theMuonLabel;

};
