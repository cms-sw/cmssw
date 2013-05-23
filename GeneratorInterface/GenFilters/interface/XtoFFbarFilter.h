#ifndef XtoFFbarFilter_h
#define XtoFFbarFilter_h

/** \class XtoFFbarFilter
 *
 *  XtoFFbarFilter
 *    A GenParticle-based filter that searches for X -> f fbar
 *    where X and f are any particle you choose.
 *    Can optionally also require a second decay Y -> g g-bar 
 *    to be present in same event.
 *
 * \author Ian Tomalin, RAL
 *
 ************************************************************/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include <vector>

class XtoFFbarFilter : public edm::EDFilter {
public:
  XtoFFbarFilter(const edm::ParameterSet&);
  ~XtoFFbarFilter() {}
  
  virtual bool filter(edm::Event& iEvent, const edm::EventSetup& iSetup);
  
private:
  
  // Check if given integer is present in vector.
  bool found(const std::vector<int>& v, int j) {
    return std::find(v.begin(), v.end(), j) != v.end();
  }
  
  // Check if given particle is X-->f fbar
  bool foundXtoFFbar(const reco::GenParticleRef& moth, 
                     const std::vector<int>& idMotherX,
		     const std::vector<int>& idDaughterF);

  virtual void endJob();

private:

  edm::InputTag src_;
  std::vector<int> idMotherX_;
  std::vector<int> idDaughterF_;
  std::vector<int> idMotherY_;
  std::vector<int> idDaughterG_;
  bool requireY_;

  edm::Handle<reco::GenParticleCollection> genParticles_;

  // For statistics only
  unsigned int xTotal_;
  double xSumPt_;
  double xSumR_;
  unsigned int totalEvents_;
  unsigned int rejectedEvents_;
};

#endif
