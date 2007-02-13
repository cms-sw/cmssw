#ifndef HLTMuonDimuonFilter_h
#define HLTMuonDimuonFilter_h

/** \class HLTMuonDimuonFilter
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing a muon pair
 *  filter for HLT muons
 *
 *  \author J. Alcaraz
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

class HLTMuonDimuonFilter : public HLTFilter {

   public:
      explicit HLTMuonDimuonFilter(const edm::ParameterSet&);
      ~HLTMuonDimuonFilter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag candTag_;  // input tag identifying product contains muons
      
      bool   fast_Accept_;      // flag to save time: stop processing after identification of the first valid pair
      double max_Eta_;          // Eta cut
      int    min_Nhits_;        // threshold on number of hits on muon
      double max_Dr_;           // impact parameter cut
      double max_Dz_;           // dz cut
      int    chargeOpt_;        // Charge option (0:nothing; +1:same charge, -1:opposite charge)
      double min_PtPair_;       // minimum Pt for the dimuon system
      double min_PtMax_;        // minimum Pt for muon with max Pt in pair
      double min_PtMin_;        // minimum Pt for muon with min Pt in pair
      double min_InvMass_;      // minimum invariant mass of pair
      double max_InvMass_;      // maximum invariant mass of pair
      double min_Acop_;         // minimum acoplanarity
      double max_Acop_;         // maximum acoplanarity
      double nsigma_Pt_;        // pt uncertainty margin (in number of sigmas)

};

#endif //HLTMuonDimuonFilter_h
