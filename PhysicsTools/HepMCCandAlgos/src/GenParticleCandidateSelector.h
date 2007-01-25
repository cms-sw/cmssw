#ifndef HepMCCandAlgos_GenParticleCandidateSelector_h
#define HepMCCandAlgos_GenParticleCandidateSelector_h
/** class 
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: GenParticleCandidateSelector.h,v 1.1 2006/11/07 12:54:02 llista Exp $
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
  /// name of particles in include or exclude list
  vstring pNameList_;
  /// using include list?
  bool bInclude_;
  /// output string for debug
  std::string caseString_;
  /// set of excluded particle id's
  std::set<int> pIds_;
  /// verbose flag
  bool verbose_;
};

#endif
