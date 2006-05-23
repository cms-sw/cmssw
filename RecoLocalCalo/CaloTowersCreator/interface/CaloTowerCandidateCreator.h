#ifndef CaloTowersCreator_CaloTowerCandidateCreator_h
#define CaloTowersCreator_CaloTowerCandidateCreator_h
/** \class CaloTowerCandidateCreator
 *
 * Framework module that produces a collection
 * of candidates with a CaloTowerCandidate compoment
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 * $Id: CaloTowerCandidateCreator.h,v 1.1 2006/04/08 00:47:57 fedor Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include <string>

namespace edm {
  class ParameterSet;
}

class CaloTowerCandidateCreator : public edm::EDProducer {
 public:
  /// constructor from parameter set
  CaloTowerCandidateCreator( const edm::ParameterSet & );
  /// destructor
  ~CaloTowerCandidateCreator();

 private:
  /// process one event
  void produce( edm::Event& e, const edm::EventSetup& );
  /// verbosity
  int mVerbose;
  /// label of source collection
  std::string mSource;
  /// ET threshold
  double mEtThreshold;
  /// E threshold
  double mEThreshold;
  /// pT threshold
  double mPtThreshold;
};

#endif
