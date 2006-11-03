#ifndef HepMCCandAlgos_GenParticleCandidateProducer_h
#define HepMCCandAlgos_GenParticleCandidateProducer_h
/** class 
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: GenParticleCandidateProducer.h,v 1.4 2006/11/02 20:41:33 llista Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidateFwd.h"
#include <vector>
#include <set>

namespace edm { class ParameterSet; }
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
  std::string src_;
  // selects only stable particles (HEPEVT status = 1)
  bool stableOnly_;
  /// exclude list
  vstring excludeList_;
  /// set of excluded particle id's
  std::set<int> excludedIds_;
  /// minimum pt
  double ptMinNeutral_, ptMinCharged_;
  /// keep initial protons
  bool keepInitialParticles_;
  /// verbose flag
  bool verbose_;
  /// pointer map type
  typedef std::map<const HepMC::GenParticle *, std::pair<int, reco::GenParticleCandidate*> > PtrMap;
  /// pointer map
  mutable PtrMap ptrMap_;
  /// reference to candidate collection
  mutable reco::GenParticleCandidateRefProd ref_;
};

#endif
