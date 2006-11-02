#ifndef HepMCCandAlgo_ParticleTreeDrawer_h
#define HepMCCandAlgo_ParticleTreeDrawer_h
/* class ParticleTreeDrawer
 *
 * \author Luca Lista, INFN
 */
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include <deque>

namespace reco { class GenParticleCandidate; }

class ParticleTreeDrawer : public edm::EDAnalyzer {
public:
  ParticleTreeDrawer( const edm::ParameterSet & );
private:
  void analyze( const edm::Event &, const edm::EventSetup & );
  edm::InputTag src_;
  std::deque<std::string> decay( const reco::GenParticleCandidate & ) const;
  edm::ESHandle<DefaultConfig::ParticleDataTable> pdt_;
};

#endif
