#ifndef HLTHtMhtFilter_h_
#define HLTHtMhtFilter_h_

/** \class HLTHtMhtFilter
 *
 *  \brief  This filters events based on HT and MHT produced by HLTHtMhtProducer2
 *  \author Steven Lowette
 *  \author Michele de Gruttola, Jia Fu Low (Nov 2013)
 *
 *  This filter can accept more than one pair of HT and MHT. An event is kept
 *  if at least one pair satisfies:
 *    - HT > `minHt_[i]` ; and
 *    - MHT > `minMht_[i]` ; and
 *    - sqrt(MHT + `meffSlope_[i]` * HT) > `minMeff_[i]`
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"


namespace edm {
    class ConfigurationDescriptions;
}

// Class declaration
class HLTHtMhtFilter : public HLTFilter {
  public:
    explicit HLTHtMhtFilter(const edm::ParameterSet & iConfig);
    ~HLTHtMhtFilter();
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    virtual bool hltFilter(edm::Event & iEvent, const edm::EventSetup & iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

  private:
    /// Minimum HT requirements
    std::vector<double> minHt_;

    /// Minimum MHT requirements
    std::vector<double> minMht_;

    /// Minimum Meff requirements
    std::vector<double> minMeff_;

    /// Meff slope requirements
    std::vector<double> meffSlope_;

    /// Input reco::MET collections to retrieve HT and MHT
    std::vector<edm::InputTag> htLabels_;
    std::vector<edm::InputTag> mhtLabels_;

    unsigned int nOrs_;  /// number of pairs of HT and MHT

    std::vector<edm::EDGetTokenT<reco::METCollection> > m_theHtToken;
    std::vector<edm::EDGetTokenT<reco::METCollection> > m_theMhtToken;
};

#endif  // HLTHtMhtFilter_h_

