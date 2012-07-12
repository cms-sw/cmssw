#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
namespace reco 
{
class PFCandidate;
}
class CaloRecHit;

class FWPFCandidateWithHitsProxyBuilder : public FWProxyBuilderBase
{
public:
   FWPFCandidateWithHitsProxyBuilder() {}
   virtual ~FWPFCandidateWithHitsProxyBuilder(){}

   virtual void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*);
   
   virtual bool havePerViewProduct(FWViewType::EType) const { return true; }

   virtual void scaleProduct(TEveElementList* parent, FWViewType::EType type, const FWViewContext* vc);

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWPFCandidateWithHitsProxyBuilder( const FWPFCandidateWithHitsProxyBuilder& );                    // Stop default
   const FWPFCandidateWithHitsProxyBuilder& operator=( const FWPFCandidateWithHitsProxyBuilder& );   // Stop default

   void addHitsForCandidate(const reco::PFCandidate& c, TEveElement* holder, const FWViewContext* vc);
   void initCaloRecHitsCollections();
   const CaloRecHit* getHitForDetId(uint32_t detId);
   void viewContextBoxScale( const float* corners, float scale, bool plotEt, std::vector<float>& scaledCorners, const CaloRecHit*);

   const HBHERecHitCollection *m_collectionHBHE;
};
