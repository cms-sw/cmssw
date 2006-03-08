#ifndef HepMCCandAlgos_HepMCCandidateProducer_h
#define HepMCCandAlgos_HepMCCandidateProducer_h
/** class 
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: HepMCCandidateProducer.h,v 1.2 2005/12/11 19:02:18 llista Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include <string>

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
  /// process one event
  void produce( edm::Event& e, const edm::EventSetup& );
  /// source collection name  
  std::string source;
  // selects only stable particles (HEPEVT status = 1)
  bool stableOnly;
};

#endif
