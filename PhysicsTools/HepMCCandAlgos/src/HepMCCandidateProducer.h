#ifndef HepMCCandAlgos_HepMCCandidateProducer_h
#define HepMCCandAlgos_HepMCCandidateProducer_h
/** class 
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: HepMCCandidateProducer.h,v 1.3 2006/03/13 18:40:00 llista Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include <string>
#include <vector>
#include <set>

namespace edm {
  class ParameterSet;
}

class HepMCCandidateProducer : public edm::EDProducer {
 public:
  /// constructor
  HepMCCandidateProducer( const edm::ParameterSet & );
  /// destructor
  ~HepMCCandidateProducer();

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
};

#endif
