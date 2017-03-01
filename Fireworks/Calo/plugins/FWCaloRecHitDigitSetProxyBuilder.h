#ifndef Fireworks_Calo_FWCaloRecHitDigitSetProxyBuilder_h
#define Fireworks_Calo_FWCaloRecHitDigitSetProxyBuilder_h

#include "TEveVector.h"
#include "Fireworks/Core/interface/FWDigitSetProxyBuilder.h"

class CaloRecHit;

class FWCaloRecHitDigitSetProxyBuilder : public FWDigitSetProxyBuilder
{
public:
   FWCaloRecHitDigitSetProxyBuilder();
   virtual ~FWCaloRecHitDigitSetProxyBuilder( void ) {}

   virtual void    setItem(const FWEventItem* iItem);

   virtual bool havePerViewProduct(FWViewType::EType) const { return true; }
   virtual void scaleProduct(TEveElementList* parent, FWViewType::EType, const FWViewContext* vc);
   virtual void build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* );
   
   virtual float scaleFactor(const FWViewContext* vc);
   virtual void  invertBox(bool x ) { m_invertBox = x ;}
   virtual void  viewContextBoxScale( const float* corners, float scale, bool plotEt, std::vector<float>& scaledCorners, const CaloRecHit*);

private:

   FWCaloRecHitDigitSetProxyBuilder( const FWCaloRecHitDigitSetProxyBuilder& );
   const FWCaloRecHitDigitSetProxyBuilder& operator=( const FWCaloRecHitDigitSetProxyBuilder& );

   bool m_invertBox;
   bool m_ignoreGeoShapeSize;
   double m_enlarge;
   TEveVector m_vector; // internal memeber, to avoid constant recreation
};
#endif
