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
#include "FWCore/ParameterSet/interface/InputTag.h"

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

  //note, all loose cuts should be looser than tight cuts
  float photonLooseMinPt;
  float photonTightMinPt;
  float photonLooseMaxEta;
  float photonTightMaxEta;
  float photonLooseMaxHoE;
  float photonTightMaxHoE;
  float photonLooseMaxHIsol;
  float photonTightMaxHIsol;
  float photonLooseMaxEIsol;
  float photonTightMaxEIsol;
  float photonLooseMaxTIsol;
  float photonTightMaxTIsol;

  //float photon2MinPt;

  int nPhotonLooseMin;
  int nPhotonTightMin;

  // Reco samples
  edm::InputTag thePhotonLabel;
  // std::string thePhotonLabel;
};

#endif
