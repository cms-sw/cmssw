/*HLTTauRefCombiner
Producer that Combines LV collection to common denominator objects
for matching in the Tau HLT

*/

#ifndef HLTTauRefCombiner_h
#define HLTTauRefCombiner_h

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <string>
#include <vector>

class HLTTauRefCombiner : public edm::global::EDProducer<> {
public:
  explicit HLTTauRefCombiner(const edm::ParameterSet &);

  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

private:
  typedef math::XYZTLorentzVectorD LorentzVector;
  typedef std::vector<LorentzVector> LorentzVectorCollection;

  std::vector<edm::EDGetTokenT<LorentzVectorCollection>> inputColl_;  // Input LV Collections
  const double matchDeltaR_;                                          // Delta R for matching
  const std::string outName_;                                         // outputObjectName

  bool match(const LorentzVector &,
             const LorentzVectorCollection &) const;  // See if this Jet Is Matched
};

#endif
