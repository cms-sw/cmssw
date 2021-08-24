#include "Fireworks/Calo/plugins/FWCaloRecHitDigitSetProxyBuilder.h"
#include "DataFormats/FTLRecHit/interface/FTLRecHitCollections.h"

class FWFTLRecHitProxyBuilder : public FWCaloRecHitDigitSetProxyBuilder {
public:
  FWFTLRecHitProxyBuilder(void) { invertBox(true); }
  ~FWFTLRecHitProxyBuilder(void) override {}

  REGISTER_PROXYBUILDER_METHODS();

  FWFTLRecHitProxyBuilder(const FWFTLRecHitProxyBuilder&) = delete;
  const FWFTLRecHitProxyBuilder& operator=(const FWFTLRecHitProxyBuilder&) = delete;
};

REGISTER_FWPROXYBUILDER(FWFTLRecHitProxyBuilder, FTLRecHitCollection, "FTL RecHit", FWViewType::kISpyBit);
