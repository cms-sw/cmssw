#ifndef CaloTowersCreator_CaloTowerCandidateCreator_h
#define CaloTowersCreator_CaloTowerCandidateCreator_h

/** \class CaloTowerCandidateCreator
 *
 * Framework module that produces a collection
 * of candidates with a CaloTowerCandidate compoment
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.4 $
 *
 * $Id: CaloTowerCandidateCreator.h,v 1.4 2006/08/17 15:23:34 mansj Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <string>


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
  edm::InputTag mSource;
  /// ET threshold
  double mEtThreshold;
  /// E threshold
  double mEThreshold;
};

#endif
