#ifndef HLTMuonL1Filter_h
#define HLTMuonL1Filter_h

/** \class HLTMuonL1Filter
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing a 
 *  filter on L1 GMT input
 *
 *  \author J. Alcaraz
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"

class HLTMuonL1Filter : public HLTFilter {

   public:
      explicit HLTMuonL1Filter(const edm::ParameterSet&);
      ~HLTMuonL1Filter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      bool triggerByPreviousLevel(const l1extra::L1MuonParticleRef &, const std::vector<l1extra::L1MuonParticleRef> &);
   private:
      edm::InputTag candTag_;  // input tag identifying product contains muons
      edm::InputTag previousCandTag_;  // input tag identifying product contains muons passing the previous level
      double max_Eta_;      // Max eta cut
      double min_Pt_;      // Pt threshold
      int min_Quality_;      // Cut on quality (probably not what we want)
      /// Quality codes:
      ///
      /// 0 .. no muon
      /// 1 .. beam halo muon (CSC)
      /// 2 .. very low quality level 1 (e.g. ignore in single and di-muon trigger)
      /// 3 .. very low quality level 2 (e.g. ignore in single muon trigger use in di-muon trigger)
      /// 4 .. very low quality level 3 (e.g. ignore in di-muon trigger, use in single-muon trigger)
      /// 5 .. unmatched RPC
      /// 6 .. unmatched DT or CSC
      /// 7 .. matched DT-RPC or CSC-RPC
      ///
      /// attention: try not to rely on quality codes in analysis: they may change again
      ///
      int min_N_;      // Minimum number of muons to pass the filter
      bool saveTag_;            // should we save the input collection ?
};

#endif //HLTMuonL1Filter_h
