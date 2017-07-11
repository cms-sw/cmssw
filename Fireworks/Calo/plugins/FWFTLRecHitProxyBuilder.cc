#include "Fireworks/Calo/plugins/FWCaloRecHitDigitSetProxyBuilder.h"
#include "DataFormats/FTLRecHit/interface/FTLRecHitCollections.h"

class FWFTLRecHitProxyBuilder : public FWCaloRecHitDigitSetProxyBuilder
{
public:
   FWFTLRecHitProxyBuilder( ) { invertBox(true); }
   virtual ~FWFTLRecHitProxyBuilder( ) {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWFTLRecHitProxyBuilder( const FWFTLRecHitProxyBuilder& );
   const FWFTLRecHitProxyBuilder& operator=( const FWFTLRecHitProxyBuilder& );
};

REGISTER_FWPROXYBUILDER( FWFTLRecHitProxyBuilder, FTLRecHitCollection, "FTL RecHit", FWViewType::kISpyBit );
