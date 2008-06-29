#ifndef PhysicsTools_HepMCCandAlgos_interface_FlavorHistoryProducer_h
#define PhysicsTools_HepMCCandAlgos_interface_FlavorHistoryProducer_h

/** class 
 *
 * \author Stephen Mrenna, FNAL
 *
 * \version $Id: FlavorHistoryProducer.cc,v 1.0
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include <string>
#include <vector>
#include <set>
#include <utility>
#include <algorithm>

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/ShallowClonePtrCandidate.h"
#include "DataFormats/HepMCCandidate/interface/FlavorHistory.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <fstream>



class FlavorHistoryProducer : public edm::EDProducer {
 public:
  /// constructor
  FlavorHistoryProducer( const edm::ParameterSet & );
  /// destructor
  ~FlavorHistoryProducer();

 private:
  /// module init at begin of job
  void beginJob( const edm::EventSetup & );
  /// process one event
  void produce( edm::Event& e, const edm::EventSetup& );

  void getAncestors(const reco::Candidate &c,
		    std::vector<reco::Candidate const * > & moms );

  
  
  edm::InputTag src_;               // source collection name 
  int    pdgIdToSelect_;            // pdg of hf partons to select
  double ptMinParticle_;            // minimum pt of the partons
  double ptMinShower_;              // minimum pt of the shower
  double etaMaxParticle_;           // max eta of the parton
  double etaMaxShower_;             // max eta of the shower
  std::string flavorHistoryName_;   // name to give flavor history
  bool verbose_;                    // verbose flag
};

#endif
