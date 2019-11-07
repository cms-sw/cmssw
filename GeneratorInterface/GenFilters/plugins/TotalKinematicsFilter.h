#ifndef TotalKinematicsFilter_h
#define TotalKinematicsFilter_h

// system include files
#include <memory>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Candidate/interface/Particle.h"

/** Filter to select events with deviation from nominal total
    kinematics smaller than a tolerance parameter. 

    This filter adds up the momenta of all generated particles 
    (except the initial state protons) and requires that the sum of the
    of the three-momenta is zero (within tolerance) and that the sum 
    of the energies is equal to the center of mass energy (within
    tolerance). The center of mass energy is obtained from the
    initial state protons. The tolerance is specified in GeV.
 */
class TotalKinematicsFilter : public edm::EDFilter {
public:
  explicit TotalKinematicsFilter(const edm::ParameterSet&);
  ~TotalKinematicsFilter() override;

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------

  edm::InputTag src_;

  /** tolerance in GeV */
  double tolerance_;

  bool verbose_;
};

#endif
