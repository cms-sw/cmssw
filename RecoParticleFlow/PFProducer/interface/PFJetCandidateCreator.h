#ifndef PFJetCandidateCreator_h
#define PFJetCandidateCreator_h

/** \class PFJetCandidateCreator
 *
 * Framework module that produces a collection
 * of candidates <edm::OwnVector<Candidate> from PFCandidates
 *
 * \author Joanna Weng , ETH Zuerich
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <string>


class PFJetCandidateCreator : public edm::EDProducer {
 public:
  /// constructor from parameter set
  PFJetCandidateCreator( const edm::ParameterSet & );
  /// destructor
  ~PFJetCandidateCreator();

 private:
  /// process one event
  void produce( edm::Event& e, const edm::EventSetup& );
  /// verbosity
  bool mVerbose;
  /// label of source collection
  edm::InputTag mSource;
};

#endif
