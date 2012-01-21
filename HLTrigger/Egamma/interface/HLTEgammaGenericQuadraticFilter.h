#ifndef HLTEgammaGenericQuadraticFilter_h
#define HLTEgammaGenericQuadraticFilter_h

/** \class HLTEgammaGenericQuadraticFilter
 *
 *  \author Roberto Covarelli (CERN)
 *  modified by Chris Tully (Princeton)
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class declaration
//

class HLTEgammaGenericQuadraticFilter : public HLTFilter {

   public:
      explicit HLTEgammaGenericQuadraticFilter(const edm::ParameterSet&);
      ~HLTEgammaGenericQuadraticFilter();
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

   private:
      edm::InputTag candTag_; // input tag identifying product that contains filtered photons
      edm::InputTag isoTag_; // input tag identifying product that contains isolated map
      edm::InputTag nonIsoTag_; // input tag identifying product that contains non-isolated map
      bool lessThan_;           // the cut is "<" or ">" ?
      bool useEt_;              // use E or Et in relative isolation cuts
/*  Barrel quadratic threshold function:
      vali (<= or >=) thrRegularEB_ + (E or Et)*thrOverEEB_ + (E or Et)*(E or Et)*thrOverE2EB_
    Endcap quadratic threshold function:
      vali (<= or >=) thrRegularEE_ + (E or Et)*thrOverEEE_ + (E or Et)*(E or Et)*thrOverE2EE_
*/
      double thrRegularEB_;     // threshold value for zeroth order term - ECAL barrel 
      double thrRegularEE_;     // threshold value for zeroth order term - ECAL endcap
      double thrOverEEB_;       // coefficient for first order term - ECAL barrel 
      double thrOverEEE_;       // coefficient for first order term - ECAL endcap 
      double thrOverE2EB_;      // coefficient for second order term - ECAL barrel 
      double thrOverE2EE_;      // coefficient for second order term - ECAL endcap 
      int    ncandcut_;        // number of photons required
      bool doIsolated_;

      edm::InputTag L1IsoCollTag_; 
      edm::InputTag L1NonIsoCollTag_; 
};

#endif //HLTEgammaGenericQuadraticFilter_h


