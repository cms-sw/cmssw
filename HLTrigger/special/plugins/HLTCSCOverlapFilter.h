#ifndef HLTCSCOverlapFilter_h
#define HLTCSCOverlapFilter_h

#include <vector>
#include <map>

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"
#include "TH1F.h"

namespace edm {
  class ConfigurationDescriptions;
}

class HLTCSCOverlapFilter : public HLTFilter {
public:
  explicit HLTCSCOverlapFilter(const edm::ParameterSet &);
  ~HLTCSCOverlapFilter() override;
  bool hltFilter(edm::Event &,
                 const edm::EventSetup &,
                 trigger::TriggerFilterObjectWithRefs &filterproduct) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  edm::InputTag m_input;
  edm::EDGetTokenT<CSCRecHit2DCollection> cscrechitsToken;
  unsigned int m_minHits;
  double m_xWindow, m_yWindow;
  bool m_ring1, m_ring2;
  bool m_fillHists;
  TH1F *m_nhitsNoWindowCut, *m_xdiff, *m_ydiff, *m_pairsWithWindowCut;
};

#endif
