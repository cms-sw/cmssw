#ifndef HepMCCandAlgos_GenParticleCandidateProducer_h
#define HepMCCandAlgos_GenParticleCandidateProducer_h
/** class 
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: GenParticleCandidateProducer.h,v 1.11 2007/01/15 12:34:59 llista Exp $
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
  /// minimum thresholds
  double ptMinNeutral_, ptMinCharged_, ptMinGluon_;
  /// keep initial protons
  bool keepInitialProtons_;
  /// suppress unfragmented partons (status=3) clones
  bool excludeUnfragmentedClones_;
  /// internal functional decomposition
  void fillIndices( const HepMC::GenEvent *, 
	     std::vector<const HepMC::GenParticle *> &, 
	     std::vector<int> &, std::vector<std::vector<int> > & ) const;
  /// internal functional decomposition
  void fillVector( const HepMC::GenEvent *,
		   std::vector<const HepMC::GenParticle *> &
		   ) const;
  /// internal functional decomposition
  void fillMothers( const std::vector<const HepMC::GenParticle *> &,
		    std::vector<int> & ) const;
  /// internal functional decomposition
  void fillDaughters( const std::vector<int> &, std::vector<std::vector<int> > & ) const;
  /// internal functional decomposition
  size_t fillSkip( const std::vector<const HepMC::GenParticle *> &, 
		   const std::vector<int> &, std::vector<bool> & ) const;
  /// internal functional decomposition
  void fix( const std::vector<const HepMC::GenParticle *> &,
	    const std::vector<int> &,
	    const std::vector<std::vector<int> > &,
	    std::vector<bool> & ) const;
  /// internal functional decomposition
  void fillOutput( const std::vector<const HepMC::GenParticle *> &,
	     const std::vector<bool> &,
	     reco::CandidateCollection &,
	     std::vector<std::pair<reco::GenParticleCandidate *, size_t> > &,
	     std::vector<size_t> & ) const;
  /// internal functional decomposition
  void fillRefs( const std::vector<int> &,
		 const reco::CandidateRefProd,
		 const std::vector<size_t> &,
		 const std::vector<std::pair<reco::GenParticleCandidate *, size_t> > &,
		 reco::CandidateCollection & ) const;
  /// charge indices
  std::vector<int> chargeP_, chargeM_;
  std::map<int, int> chargeMap_;
  int chargeTimesThree( int ) const;
};

#endif
