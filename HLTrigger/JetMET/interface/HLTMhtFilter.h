#ifndef HLTMhtFilter_h_
#define HLTMhtFilter_h_

/** \class HLTMhtFilter
 *
 *  \brief  This filters events based on HT and MHT produced by HLTHtMhtProducer2
 *  \author Steven Lowette
 *  \author Michele de Gruttola, Jia Fu Low (Nov 2013)
 *
 *  This filter can accept more than one variant of MHT. An event is kept
 *  if at least one satisfies MHT > `minHht_[i]`.
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"


namespace edm {
    class ConfigurationDescriptions;
}

// Class declaration
class HLTMhtFilter : public HLTFilter {
  public:
    explicit HLTMhtFilter(const edm::ParameterSet & iConfig);
    ~HLTMhtFilter();
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    virtual bool hltFilter(edm::Event & iEvent, const edm::EventSetup & iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

  private:
    /// Minimum MHT requirements
    std::vector<double> minMht_;

    /// Input reco::MET collections to retrieve MHT
    std::vector<edm::InputTag> mhtLabels_;

    unsigned int nOrs_;  /// number of pairs of MHT

    std::vector<edm::EDGetTokenT<reco::METCollection> > m_theMhtToken;
};

#endif  // HLTMhtFilter_h_

