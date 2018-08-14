#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
//#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
namespace reco 
{
class PFCandidate;
class PFRecHit;
}
class CaloRecHit;

class FWPFCandidateWithHitsProxyBuilder : public FWProxyBuilderBase
{
public:
   FWPFCandidateWithHitsProxyBuilder() {}
   ~FWPFCandidateWithHitsProxyBuilder() override{}

   void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*) override;
   
   bool havePerViewProduct(FWViewType::EType) const override { return true; }

   void scaleProduct(TEveElementList* parent, FWViewType::EType type, const FWViewContext* vc) override;

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWPFCandidateWithHitsProxyBuilder( const FWPFCandidateWithHitsProxyBuilder& ) = delete;                    // Stop default
   const FWPFCandidateWithHitsProxyBuilder& operator=( const FWPFCandidateWithHitsProxyBuilder& ) = delete;   // Stop default

   void addHitsForCandidate(const reco::PFCandidate& c, TEveElement* holder, const FWViewContext* vc);
   void initPFRecHitsCollections();
   const reco::PFRecHit* getHitForDetId(unsigned detId);
   void viewContextBoxScale( const float* corners, float scale, bool plotEt, std::vector<float>& scaledCorners, const reco::PFRecHit*);

   const reco::PFRecHitCollection *m_collectionHCAL;

};
