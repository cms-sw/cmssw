#ifndef HLTCaloJetTimingFilter_h_
#define HLTCaloJetTimingFilter_h_

/** \class HLTCaloJetTimingFilter
 *
 *  \brief  This makes selections on the timing and associated ecal cells 
 *  produced by HLTCaloJetTimingProducer
 *  \author Matthew Citron
 *
 *
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//
class HLTCaloJetTimingFilter : public HLTFilter {
public:
  explicit HLTCaloJetTimingFilter(const edm::ParameterSet& iConfig);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

private:
  //Input collections
  edm::InputTag jetLabel_;
  edm::InputTag jetTimeLabel_;
  edm::InputTag jetCellsForTimingLabel_;
  edm::InputTag jetEcalEtForTimingLabel_;
  //Thresholds for selection
  unsigned int minJets_;
  double jetTimeThresh_;
  double jetEcalEtForTimingThresh_;
  unsigned int jetCellsForTimingThresh_;
  double minPt_;

  edm::EDGetTokenT<reco::CaloJetCollection> jetInputToken;
  edm::EDGetTokenT<edm::ValueMap<float>> jetTimesInputToken;
  edm::EDGetTokenT<edm::ValueMap<unsigned int>> jetCellsForTimingInputToken;
  edm::EDGetTokenT<edm::ValueMap<float>> jetEcalEtForTimingInputToken;
};
#endif  // HLTCaloJetTimingFilter_h_
