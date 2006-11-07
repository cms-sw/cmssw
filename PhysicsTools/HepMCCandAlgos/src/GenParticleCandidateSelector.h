#ifndef HepMCCandAlgos_GenParticleCandidateSelector_h
#define HepMCCandAlgos_GenParticleCandidateSelector_h
/** class 
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: GenParticleCandidateSelector.h,v 1.5 2006/10/06 12:06:59 llista Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include <string>
#include <vector>
#include <set>

class GenParticleCandidateSelector : public edm::EDProducer {
 public:
  /// constructor
  GenParticleCandidateSelector( const edm::ParameterSet & );
  /// destructor
  ~GenParticleCandidateSelector();

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
  /// verbose flag
  bool verbose_;
};

#endif
