#ifndef CaloTowerCreator_CaloTowerFromL1TCreatorForTauHLT_h
#define CaloTowerCreator_CaloTowerFromL1TCreatorForTauHLT_h

/** \class CaloTowerFromL1TCreatorForTauHLT
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

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDefs.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include <string>

namespace edm {
  class ParameterSet;
}

class CaloTowerFromL1TCreatorForTauHLT : public edm::global::EDProducer<> {
public:
  /// constructor from parameter set
  CaloTowerFromL1TCreatorForTauHLT(const edm::ParameterSet&);
  /// destructor
  ~CaloTowerFromL1TCreatorForTauHLT() override = default;
  ///
  static void fillDescriptions(edm::ConfigurationDescriptions& desc);

private:
  /// process one event
  void produce(edm::StreamID sid, edm::Event& evt, const edm::EventSetup& stp) const override;

  /// bunch crossing
  const int mBX;
  /// verbosity
  const int mVerbose;
  /// label of source collection
  const edm::EDGetTokenT<CaloTowerCollection> mtowers_token;
  /// use only towers in cone mCone around L1 candidate for regional jet reco
  const double mCone, mCone2;
  /// label of tau trigger type analysis
  const edm::EDGetTokenT<l1t::TauBxCollection> mTauTrigger_token;
  /// ET threshold
  const double mEtThreshold;
  /// E threshold
  const double mEThreshold;
  //
  const int mTauId;
};

#endif
