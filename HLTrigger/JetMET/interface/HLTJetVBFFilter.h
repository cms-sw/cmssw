#ifndef HLTJetVBFFilter_h
#define HLTJetVBFFilter_h

/** \class HLTJetVBFFilter
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class decleration
//

class HLTJetVBFFilter : public HLTFilter {

   public:
      explicit HLTJetVBFFilter(const edm::ParameterSet&);
      ~HLTJetVBFFilter();
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

   private:
      edm::InputTag inputTag_; // input tag identifying jets
      double minEtLow_;
      double minEtHigh_;
      bool etaOpposite_;
      double minDeltaEta_;
      double minInvMass_;
      double maxEta_;
};

#endif //HLTJetVBFFilter_h
