#ifndef HLTEgammaGenericFilter_h
#define HLTEgammaGenericFilter_h

/** \class HLTEgammaGenericFilter
 *
 *  \author Roberto Covarelli (CERN)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class declaration
//

class HLTEgammaGenericFilter : public HLTFilter {

   public:
      explicit HLTEgammaGenericFilter(const edm::ParameterSet&);
      ~HLTEgammaGenericFilter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag candTag_; // input tag identifying product that contains filtered photons
      edm::InputTag isoTag_; // input tag identifying product that contains isolated map
      edm::InputTag nonIsoTag_; // input tag identifying product that contains non-isolated map
      bool lessThan_;           // the cut is "<" or ">" ?
      bool useEt_;              // use E or Et in relative isolation cuts
      double thrRegularEB_;     // threshold for regular cut (x < thr) - ECAL barrel 
      double thrRegularEE_;     // threshold for regular cut (x < thr) - ECAL endcap
      double thrOverEEB_;       // threshold for x/E < thr cut (isolations) - ECAL barrel 
      double thrOverEEE_;       // threshold for x/E < thr cut (isolations) - ECAL endcap 
      double thrOverE2EB_;      // threshold for x/E^2 < thr cut (isolations) - ECAL barrel 
      double thrOverE2EE_;      // threshold for x/E^2 < thr cut (isolations) - ECAL endcap 
      int    ncandcut_;        // number of photons required
      bool doIsolated_;

      bool   store_;
      edm::InputTag L1IsoCollTag_; 
      edm::InputTag L1NonIsoCollTag_; 
};

#endif //HLTEgammaGenericFilter_h


