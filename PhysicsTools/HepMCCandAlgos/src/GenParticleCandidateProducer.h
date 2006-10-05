#ifndef HepMCCandAlgos_GenParticleCandidateProducer_h
#define HepMCCandAlgos_GenParticleCandidateProducer_h
/** class 
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: GenParticleCandidateProducer.h,v 1.4 2006/04/12 07:33:17 llista Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include <string>
#include <vector>
#include <set>

namespace edm { class ParameterSet; }
namespace reco { class GenParticleCandidate; }
namespace HepMC { class GenParticle; }

class GenParticleCandidateProducer : public edm::EDProducer {
 public:
  /// constructor
  GenParticleCandidateProducer( const edm::ParameterSet & );
  /// destructor
  ~GenParticleCandidateProducer();

 private:
  /// vector of strings
  typedef std::vector<std::string> vstring;
  /// module init at begin of job
  void beginJob( const edm::EventSetup & );
  /// process one event
  void produce( edm::Event& e, const edm::EventSetup& );
  /// source collection name  
  std::string source;
  // selects only stable particles (HEPEVT status = 1)
  bool stableOnly;
  /// exclude list
  vstring excludeList;
  /// set of excluded particle id's
  std::set<int> excludedIds;
  /// verbose flag
  bool verbose;
  /// pointer map type
  typedef std::map<const HepMC::GenParticle *, std::pair<int, reco::GenParticleCandidate*> > PtrMap;
  /// add daughters to a candidate
  void addDaughters( reco::GenParticleCandidate *, const HepMC::GenParticle * ) const;
  /// pointer map
  mutable PtrMap ptrMap_;
  /// reference to candidate collection
  mutable reco::CandidateRefProd ref_;
};

#endif
