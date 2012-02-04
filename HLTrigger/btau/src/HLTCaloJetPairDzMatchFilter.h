#ifndef HLTCaloJetPairDzMatchFilter_h
#define HLTCaloJetPairDzMatchFilter_h

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/** class HLTCaloJetPairDzMatchFilter
 * an HLT filter which picks up a CaloJetCollection (supposedly, of L2 tau jets)
 * and passes only events with at least one pair of non-overlapping jets with 
 * vertices within some dz
 */

class HLTCaloJetPairDzMatchFilter : public HLTFilter {

  public:

    explicit HLTCaloJetPairDzMatchFilter(const edm::ParameterSet&);
    ~HLTCaloJetPairDzMatchFilter();
    virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);
    
  private:

    edm::InputTag m_jetSrc;
    double m_jetMinPt;
    double m_jetMaxEta;
    double m_jetMinDR;
    double m_jetMaxDZ;

};

#endif 
