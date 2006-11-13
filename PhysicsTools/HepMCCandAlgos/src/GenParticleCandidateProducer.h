#ifndef HepMCCandAlgos_GenParticleCandidateProducer_h
#define HepMCCandAlgos_GenParticleCandidateProducer_h
/** class 
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: GenParticleCandidateProducer.h,v 1.7 2006/11/13 12:43:49 llista Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
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
  double eMinNeutral_, ptMinCharged_;
  /// keep initial protons
  bool keepInitialProtons_;
  /// suppress unfragmented partons (status=3) clones
  bool excludeUnfragmentedClones_;
};

#endif
