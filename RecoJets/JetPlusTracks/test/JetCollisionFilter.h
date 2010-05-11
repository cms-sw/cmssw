#ifndef JetPlusTrackCollisionAnalysis_h
#define JetPlusTrackCollisionAnalysis_h
// user include files
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <string>
#include <memory>
#include <map>

namespace edm {
  class Event;
  class EventSetup;
  class ParameterSet;
}

///
/// jet energy corrections from MCjet calibration
///
namespace cms
{

class JetCollisionFilter : public edm::EDFilter
{
public:  

  JetCollisionFilter(const edm::ParameterSet& fParameters);

  virtual ~JetCollisionFilter();

  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void beginJob() ;
  virtual void endJob() ;
   
private:
// Histograms/Tree

  bool allowMissingInputs_;

  edm::InputTag mInputJets;
 
};
}
#endif
