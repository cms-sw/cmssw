#ifndef HLTJetPairDzMatchFilter_h
#define HLTJetPairDzMatchFilter_h

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/** class HLTJetPairDzMatchFilter
 * an HLT filter which picks up a JetCollection (supposedly, of L2 tau jets)
 * and passes only events with at least one pair of non-overlapping jets with
 * vertices within some dz
 */
namespace edm {
  class ConfigurationDescriptions;
}

template<typename T>
class HLTJetPairDzMatchFilter : public HLTFilter {

  public:

    explicit HLTJetPairDzMatchFilter(const edm::ParameterSet&);
    ~HLTJetPairDzMatchFilter();
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

  private:

    edm::InputTag                     m_jetTag;
    edm::EDGetTokenT<std::vector<T> > m_jetToken;
    double m_jetMinPt;
    double m_jetMaxEta;
    double m_jetMinDR;
    double m_jetMaxDZ;
    int    m_triggerType;

};

#endif
