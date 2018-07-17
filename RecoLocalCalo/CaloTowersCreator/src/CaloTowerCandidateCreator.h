#ifndef CaloTowersCreator_CaloTowerCandidateCreator_h
#define CaloTowersCreator_CaloTowerCandidateCreator_h

/** \class CaloTowerCandidateCreator
 *
 * Framework module that produces a collection
 * of candidates with a CaloTowerCandidate compoment
 *
 * \author Luca Lista, INFN
 *
 *
 *
 */
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include <string>


class CaloTowerCandidateCreator : public edm::stream::EDProducer<> {
 public:
  /// constructor from parameter set
  CaloTowerCandidateCreator( const edm::ParameterSet & );
  /// destructor
  ~CaloTowerCandidateCreator() override;

 private:
  /// process one event
  void produce( edm::Event& e, const edm::EventSetup& ) override;
  /// verbosity
  int mVerbose;
  /// token of source collection
  edm::EDGetTokenT<CaloTowerCollection> tok_src_;
  /// ET threshold
  double mEtThreshold;
  /// E threshold
  double mEThreshold;
};

#endif
