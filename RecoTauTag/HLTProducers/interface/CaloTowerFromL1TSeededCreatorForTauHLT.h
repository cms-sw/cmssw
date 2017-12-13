#ifndef CaloTowerCreator_CaloTowerFromL1TSeededCreatorForTauHLT_h
#define CaloTowerCreator_CaloTowerFromL1TSeededCreatorForTauHLT_h

/** \class CaloTowerFromL1TSeededCreatorForTauHLT
 *
 * Framework module that produces a collection
 * of calo towers in the region of interest for Tau HLT reconstruction,
 * defined around L1 seeds
 *
 * \author T. Strebler. IC.   based on A. Nikitenko
 *
 */

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include <string>

namespace edm {
  class ParameterSet;
}

class CaloTowerFromL1TSeededCreatorForTauHLT : public edm::global::EDProducer<> {
 public:
  /// constructor from parameter set
  CaloTowerFromL1TSeededCreatorForTauHLT( const edm::ParameterSet & );
  /// destructor
  ~CaloTowerFromL1TSeededCreatorForTauHLT();
  /// 
  static void fillDescriptions( edm::ConfigurationDescriptions& desc );

 private:
  /// process one event
  void produce( edm::StreamID sid, edm::Event& evt, const edm::EventSetup& stp ) const override;

  /// verbosity
  const int mVerbose;
  /// label of source collection
  const edm::EDGetTokenT<CaloTowerCollection> mtowers_token;
  /// use only towers in cone mCone around L1 candidate for regional jet reco
  const double mCone;
  /// label of tau trigger type analysis
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> mTauTrigger_token;
  /// imitator of L1 seeds
  //edm::InputTag ml1seeds;
  /// ET threshold
  const double mEtThreshold;
  /// E threshold
  const double mEThreshold;

};

#endif
