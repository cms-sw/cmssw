#ifndef HLTDisplacedmumuFilter_h
#define HLTDisplacedmumuFilter_h

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
namespace edm {
  class ConfigurationDescriptions;
}
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

class HLTDisplacedmumuFilter : public HLTFilter {

  public:
    explicit HLTDisplacedmumuFilter(const edm::ParameterSet&);
    ~HLTDisplacedmumuFilter();
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);	
    virtual void beginJob() ;
    virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;
    virtual void endJob() ;

  private:
    bool fastAccept_;
    double minLxySignificance_;
    double maxLxySignificance_;
    double maxNormalisedChi2_;
    double minVtxProbability_;
    double minCosinePointingAngle_;
    edm::InputTag                            DisplacedVertexTag_;
    edm::EDGetTokenT<reco::VertexCollection> DisplacedVertexToken_;
    edm::InputTag                            beamSpotTag_;
    edm::EDGetTokenT<reco::BeamSpot>         beamSpotToken_;
    edm::InputTag                                          MuonTag_;
    edm::EDGetTokenT<reco::RecoChargedCandidateCollection> MuonToken_;

};
#endif
