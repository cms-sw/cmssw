#ifndef HepMCCandAlgos_HepMCCandidateProducer_h
#define HepMCCandAlgos_HepMCCandidateProducer_h
/** class 
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: HepMCCandidateProducer.h,v 1.1 2006/03/08 10:50:07 llista Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include <string>
#include <vector>
#include <set>
#include <CLHEP/HepPDT/DefaultConfig.hh>

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
  /// process one event
  void produce( edm::Event& e, const edm::EventSetup& );
  /// source collection name  
  std::string source;
  /// pdt file name
  std::string pdtFileName;
  // selects only stable particles (HEPEVT status = 1)
  bool stableOnly;
  /// exclude list
  vstring excludeList;
  /// particle data table
  DefaultConfig::ParticleDataTable pdt;
  /// set of excluded particle id's
  std::set<int> excludedIds;
};

#endif
