#ifndef HLTEgammaGenericQuadraticEtaFilter_h
#define HLTEgammaGenericQuadraticEtaFilter_h

/** \class HLTEgammaGenericQuadraticEtaFilter
 *
 *  \author Roberto Covarelli (CERN)
 *  modified by Chris Tully (Princeton)
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class declaration
//

class HLTEgammaGenericQuadraticEtaFilter : public HLTFilter {

   public:
      explicit HLTEgammaGenericQuadraticEtaFilter(const edm::ParameterSet&);
      ~HLTEgammaGenericQuadraticEtaFilter();
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
      double etaBoundaryEB12_;     //eta Boundary between Regions 1 and 2 - ECAL barrel 
      double etaBoundaryEE12_;     //eta Boundary between Regions 1 and 2 - ECAL endcap
      double thrRegularEB1_;     // threshold value for zeroth order term - ECAL barrel region 1
      double thrRegularEE1_;     // threshold value for zeroth order term - ECAL endcap region 1
      double thrOverEEB1_;       // coefficient for first order term - ECAL barrel region 1
      double thrOverEEE1_;       // coefficient for first order term - ECAL endcap region 1
      double thrOverE2EB1_;      // coefficient for second order term - ECAL barrel region 1
      double thrOverE2EE1_;      // coefficient for second order term - ECAL endcap region 1
      double thrRegularEB2_;     // threshold value for zeroth order term - ECAL barrel region 2
      double thrRegularEE2_;     // threshold value for zeroth order term - ECAL endcap region 2
      double thrOverEEB2_;       // coefficient for first order term - ECAL barrel region 2
      double thrOverEEE2_;       // coefficient for first order term - ECAL endcap region 2
      double thrOverE2EB2_;      // coefficient for second order term - ECAL barrel region 2
      double thrOverE2EE2_;      // coefficient for second order term - ECAL endcap region 2
      int    ncandcut_;        // number of photons required
      bool doIsolated_;

      bool   store_;
      edm::InputTag L1IsoCollTag_; 
      edm::InputTag L1NonIsoCollTag_; 
};

#endif //HLTEgammaGenericQuadraticEtaFilter_h


