/** Description: Check correlation between PFJet pairs and filtered PFTau pairs and store the PFJet pairs.
For (j1, j2, t1, t2) where j1, j2 from the PFJet collection and t1, t2 from the filtered PFTau collection,
the module checks if there is no overlap (within dRmin) between j1, j2, t1, t2, i.e. they are 4 different objects.
In addition, the module imposes the following cuts:
* mjjMin: the min invariant mass cut on (j1, j2)
* extraTauPtCut: the leading tau pt cut on (t1, t2) (under the assumption t1, t2 are products of a subleading pt filter with minN = 2)
The module stores j1, j2 of any (j1, j2, t1, t2) that satisfies the conditions above. */

#ifndef RecoTauTag_HLTProducers_PFDiJetCorrCheckerWithDiTau_H
#define RecoTauTag_HLTProducers_PFDiJetCorrCheckerWithDiTau_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

class PFDiJetCorrCheckerWithDiTau : public edm::stream::EDProducer<> {
private:
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> tauSrc_;
  const edm::EDGetTokenT<reco::PFJetCollection> pfJetSrc_;
  const double extraTauPtCut_;
  const double mjjMin_;
  const double matchingR2_;

public:
  explicit PFDiJetCorrCheckerWithDiTau(const edm::ParameterSet&);
  ~PFDiJetCorrCheckerWithDiTau() override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
};
#endif
