#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

class FWJetLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Jet>
{
public:
   FWJetLegoProxyBuilder() {}
   virtual ~FWJetLegoProxyBuilder() {}
  
   REGISTER_PROXYBUILDER_METHODS();

protected:
   virtual void build(const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder,
                      const FWViewContext*);

private:
   FWJetLegoProxyBuilder( const FWJetLegoProxyBuilder& ); // stop default
   const FWJetLegoProxyBuilder& operator=( const FWJetLegoProxyBuilder& ); // stop default
};

void
FWJetLegoProxyBuilder::build(const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder,
                             const FWViewContext*) 
{
   fireworks::addCircle( iData.eta(), iData.phi(), 0.5, 20, &oItemHolder, this );
}

REGISTER_FWPROXYBUILDER( FWJetLegoProxyBuilder, reco::Jet, "Jets", FWViewType::kLegoBit | FWViewType::kLegoHFBit );
