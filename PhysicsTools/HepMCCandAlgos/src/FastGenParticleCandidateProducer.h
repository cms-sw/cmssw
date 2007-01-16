#ifndef HepMCCandAlgos_FastGenParticleCandidateProducer_h
#define HepMCCandAlgos_FastGenParticleCandidateProducer_h
/** class 
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: FastGenParticleCandidateProducer.h,v 1.1 2007/01/15 14:24:49 llista Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidateFwd.h"
#include <vector>
#include <map>
#include <set>

namespace edm { class ParameterSet; }
namespace HepMC { class GenParticle; class GenEvent; }

class FastGenParticleCandidateProducer : public edm::EDProducer {
 public:
  /// constructor
  FastGenParticleCandidateProducer( const edm::ParameterSet & );
  /// destructor
  ~FastGenParticleCandidateProducer();

 private:
  /// vector of strings
  typedef std::vector<std::string> vstring;
  /// module init at begin of job
  void beginJob( const edm::EventSetup & );
  /// process one event
  void produce( edm::Event& e, const edm::EventSetup& );
  /// source collection name  
  std::string src_;
  /// internal functional decomposition
  void fillIndices( const HepMC::GenEvent *, 
	     std::vector<const HepMC::GenParticle *> & ) const;
  /// internal functional decomposition
  void fillOutput( const std::vector<const HepMC::GenParticle *> &,
		   reco::CandidateCollection &, 
		   std::vector<reco::GenParticleCandidate *> & ) const;
  /// internal functional decomposition
  void fillRefs( const std::vector<const HepMC::GenParticle *> &,
		 const reco::CandidateRefProd,
		 const std::vector<reco::GenParticleCandidate *> & ) const;
  /// charge indices
  std::vector<int> chargeP_, chargeM_;
  std::map<int, int> chargeMap_;
  int chargeTimesThree( int ) const;
};

#endif
