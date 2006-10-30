#ifndef HepMCCandAlgos_SetGenParticleMotherReference_h
#define HepMCCandAlgos_SetGenParticleMotherReference_h
/* class SetGenParticleMotherReference
 *
 * \author Luca Lista, INFN
 */
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

class SetGenParticleMotherReference : public edm::EDAnalyzer {
public:
  SetGenParticleMotherReference( const edm::ParameterSet & );
private:
  void analyze( const edm::Event &, const edm::EventSetup & );
  edm::InputTag src_;
};

#endif
