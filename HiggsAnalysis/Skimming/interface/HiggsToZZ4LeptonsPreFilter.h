#ifndef HiggsAnalysis_HiggsToZZ4LeptonsPreFilter
#define HiggsAnalysis_HiggsToZZ4LeptonsPreFilter

/* \class HiggsTo4LeptonsSkim
 *
 *
 * Filter to select 4 lepton events (4e, 4mu, 2e2mu) within
 * fiducial volume (|eta| < 2.4)
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

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

class HiggsToZZ4LeptonsPreFilter : public edm::EDFilter {

 public:
  // Constructor
  explicit HiggsToZZ4LeptonsPreFilter(const edm::ParameterSet&);

  // Destructor
  ~HiggsToZZ4LeptonsPreFilter() override;

  /// Get event properties to send to builder to fill seed collection
  bool filter(edm::Event&, const edm::EventSetup& ) override;


 private:
  int evt, ikept;

  bool debug;
  int leptonFlavour;

  edm::EDGetTokenT<reco::GenParticleCollection> genToken;
};

#endif
