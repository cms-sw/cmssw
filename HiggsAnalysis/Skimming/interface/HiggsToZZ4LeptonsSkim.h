#ifndef HiggsAnalysis_HiggsToZZ4LeptonsSkim
#define HiggsAnalysis_HiggsToZZ4LeptonsSkim

/* \class HiggsTo4LeptonsSkim
 *
 *
 * Filter to select 4 lepton events based on the
 * 1 or 2 electron or 1 or 2 muon HLT trigger,
 * and four leptons (no flavour requirement).
 * No charge requirements are applied on event.
 *
 * \author Dominique Fortin - UC Riverside
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

#include <DataFormats/TrackReco/interface/TrackFwd.h>
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

class HiggsToZZ4LeptonsSkim : public edm::EDFilter {

 public:
  // Constructor
  explicit HiggsToZZ4LeptonsSkim(const edm::ParameterSet&);

  // Destructor
  ~HiggsToZZ4LeptonsSkim() override;

  /// Get event properties to send to builder to fill seed collection
  bool filter(edm::Event&, const edm::EventSetup& ) override;


 private:
  int nEvents, nSelectedEvents;


  bool debug;
  float stiffMinPt;
  float softMinPt;
  int nStiffLeptonMin;
  int nLeptonMin;

  // Reco samples
  edm::EDGetTokenT<reco::TrackCollection> theGLBMuonToken;
  edm::EDGetTokenT<reco::GsfElectronCollection> theGsfEToken;
};

#endif
