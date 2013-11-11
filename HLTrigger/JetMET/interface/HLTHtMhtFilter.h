#ifndef HLTHtMhtFilter_h
#define HLTHtMhtFilter_h

/** \class HLTHtMhtFilter
 *
 *  \author Steven Lowette
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/METReco/interface/MET.h"

namespace edm {
  class ConfigurationDescriptions;
}


class HLTHtMhtFilter : public HLTFilter {

  public:

    explicit HLTHtMhtFilter(const edm::ParameterSet &);
    ~HLTHtMhtFilter();
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    virtual bool hltFilter(edm::Event & iEvent, const edm::EventSetup & iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) override;

  private:
    std::vector<edm::EDGetTokenT<std::vector<reco::MET>>> m_theHtToken;
    std::vector<edm::EDGetTokenT<std::vector<reco::MET>>> m_theMhtToken;
    std::string moduleLabel_;
    std::vector<edm::InputTag> htLabels_;
    std::vector<edm::InputTag> mhtLabels_;
    std::vector<double> minHt_;
    std::vector<double> minMht_;
    std::vector<double> minMeff_;
    std::vector<double> meffSlope_;
    unsigned int nOrs_;
};

#endif
