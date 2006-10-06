#ifndef HepMCCandAlgos_HepMCCandidateProducer_h
#define HepMCCandAlgos_HepMCCandidateProducer_h
/** class 
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: HepMCCandidateProducer.h,v 1.4 2006/04/12 07:33:17 llista Exp $
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
