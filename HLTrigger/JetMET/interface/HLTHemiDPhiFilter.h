#ifndef HLTHemiDPhiFilter_h
#define HLTHemiDPhiFilter_h

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include<vector>
#include "TVector3.h"
#include "TLorentzVector.h"

namespace edm {
   class ConfigurationDescriptions;
}

//
// class declaration
//

class HLTHemiDPhiFilter : public HLTFilter {

   public:

      explicit HLTHemiDPhiFilter(const edm::ParameterSet&);
      ~HLTHemiDPhiFilter();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;


   private:
      edm::EDGetTokenT<std::vector<math::XYZTLorentzVector>> m_theHemiToken;
      static double deltaPhi(double, double); //helper function

      edm::InputTag inputTag_; // input tag identifying product
      double min_dphi_;          // minimum dphi value
      bool accept_NJ_;         // accept or reject events with high NJ

};

#endif //HLTHemiDPhiFilter_h
