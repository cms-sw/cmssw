#ifndef HiggsAnalysis_HiggsTo2GammaSkim
#define HiggsAnalysis_HiggsTo2GammaSkim

/* \class HiggsTo2GammaSkim
 *
 *
 * Filter to select 2 photon events based on the
 * 1 or 2 photon HLT trigger, 
 *
 * \author Kati Lassila-Perini - Helsinki Institute of Physics
 *
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

using namespace edm;
using namespace std;

class HiggsTo2GammaSkim : public edm::EDFilter {
  
 public:
  // Constructor
  explicit HiggsTo2GammaSkim(const edm::ParameterSet&);

  // Destructor
  ~HiggsTo2GammaSkim();

  /// Get event properties to send to builder to fill seed collection
  virtual bool filter(edm::Event&, const edm::EventSetup& );


 private:
  int nEvents, nSelectedEvents;


  bool debug;
  float photon1MinPt;
  float photon2MinPt;
  int nPhotonMin;

  // Reco samples
  edm::InputTag thePhotonLabel;
  // std::string thePhotonLabel;
};

#endif
