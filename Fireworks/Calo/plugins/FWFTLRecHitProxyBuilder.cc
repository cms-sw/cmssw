#include "Fireworks/Calo/plugins/FWCaloRecHitDigitSetProxyBuilder.h"
#include "DataFormats/FTLRecHit/interface/FTLRecHitCollections.h"

class FWFTLRecHitProxyBuilder : public FWCaloRecHitDigitSetProxyBuilder
{
public:
   FWFTLRecHitProxyBuilder( void ) { invertBox(true); }
   virtual ~FWFTLRecHitProxyBuilder( void ) {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWFTLRecHitProxyBuilder( const FWFTLRecHitProxyBuilder& );
   const FWFTLRecHitProxyBuilder& operator=( const FWFTLRecHitProxyBuilder& );
};

REGISTER_FWPROXYBUILDER( FWFTLRecHitProxyBuilder, FTLRecHitCollection, "FTL RecHit", FWViewType::kISpyBit );
