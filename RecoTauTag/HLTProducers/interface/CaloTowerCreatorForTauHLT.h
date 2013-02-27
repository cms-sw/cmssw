#ifndef CaloTowerCreator_CaloTowerCreatorForTauHLT_h
#define CaloTowerCreator_CaloTowerCreatorForTauHLT_h

/** \class CaloTowerCreatorForTauHLT
 *
 * Framework module that produces a collection
 * of calo towers in the region of interest for Tau HLT reconnstruction,
 * depending on tau type trigger:  
 *                   Tau1 - take location of 1st L1 Tau
 *                   Tau2 - take location of 2nd L1 Tau; if does not exists,
 *                          take location of 1st Calo Tower
 *                   ETau - take L1 Tau candidate which is not collinear
 *                          to HLT (or L1) electron candidate.  
 *
 * \author A. Nikitenko. IC.   based on L. Lista and J. Mans
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include <string>
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {
  class ParameterSet;
}

class CaloTowerCreatorForTauHLT : public edm::EDProducer {
 public:
  /// constructor from parameter set
  CaloTowerCreatorForTauHLT( const edm::ParameterSet & );
  /// destructor
  ~CaloTowerCreatorForTauHLT();

 private:
  /// process one event
  void produce( edm::Event& e, const edm::EventSetup& ) override;
  /// verbosity
  int mVerbose;
  /// label of source collection
 edm::InputTag mtowers;
  /// use only towers in cone mCone around L1 candidate for regional jet reco
  double mCone;
  /// label of tau trigger type analysis
  edm::InputTag mTauTrigger;
  /// imitator of L1 seeds
  edm::InputTag ml1seeds;
  /// ET threshold
  double mEtThreshold;
  /// E threshold
  double mEThreshold;

  //
  int mTauId;

};

#endif
