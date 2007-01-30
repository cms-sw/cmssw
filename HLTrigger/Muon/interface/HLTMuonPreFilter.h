#ifndef HLTMuonPreFilter_h
#define HLTMuonPreFilter_h

/** \class HLTMuonPreFilter
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing a first
 *  filtering for HLT muons
 *
 *  \author J. Alcaraz
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

class HLTMuonPreFilter : public HLTFilter {

   public:
      explicit HLTMuonPreFilter(const edm::ParameterSet&);
      ~HLTMuonPreFilter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag candTag_;  // input tag identifying product contains muons
      
      int    min_N_;            // minimum number of muons to fire the trigger
      double max_Eta_;          // Eta cut
      int    min_Nhits_;        // threshold on number of hits on muon
      double max_Dr_;           // impact parameter cut
      double max_Dz_;           // dz cut
      double min_Pt_;           // pt threshold in GeV 
      double nsigma_Pt_;        // pt uncertainty margin (in number of sigmas)
};

#endif //HLTMuonPreFilter_h
