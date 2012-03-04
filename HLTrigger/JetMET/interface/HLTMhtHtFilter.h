#ifndef HLTMhtHtFilter_h
#define HLTMhtHtFilter_h

/** \class HLTMhtHtFilter
 *
 *  \author Gheorghe Lungu
 *
 */

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

namespace edm {
   class ConfigurationDescriptions;
}

//
// class declaration
//

template<typename T>
class HLTMhtHtFilter : public HLTFilter {

   public:
      explicit HLTMhtHtFilter(const edm::ParameterSet&);
      ~HLTMhtHtFilter();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

   private:
      const edm::InputTag       inputJetTag_;   // input tag identifying jets
      const edm::InputTag       inputTracksTag_;
      const std::vector<double> minPtJet_;
      const std::vector<double> etaJet_;
      const double              minPT12_;
      const double              minHt_;
      const double              minMht_;
      const double              minAlphaT_;
      const double              minMeff_;
      const double              meffSlope_;
      const int                 minNJet_;
      const int                 mode_;          //----mode = 1 for MHT only
                                                //----mode = 2 for Meff
                                                //----mode = 3 for PT12
                                                //----mode = 4 for HT only
                                                //----mode = 5 for HT and AlphaT cross trigger (ALWAYS uses jet ET, not pT)
      const bool                usePt_;
      const bool                useTracks_; 
      int   triggerType_;
};

#endif //HLTMhtHtFilter_h
