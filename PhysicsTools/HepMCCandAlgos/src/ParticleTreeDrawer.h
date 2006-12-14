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

namespace reco { class GenParticleCandidate; }

class ParticleTreeDrawer : public edm::EDAnalyzer {
public:
  ParticleTreeDrawer( const edm::ParameterSet & );
private:
  void analyze( const edm::Event &, const edm::EventSetup & );
  edm::InputTag src_;
  void printDecay( const reco::GenParticleCandidate &, const std::string & pre ) const;
  edm::ESHandle<DefaultConfig::ParticleDataTable> pdt_;
  /// print parameters
  bool printP4_, printPtEtaPhi_, printVertex_, printStatus_;
  /// print 4 momenta
  void printP4( const reco::GenParticleCandidate & ) const;
};

#endif
