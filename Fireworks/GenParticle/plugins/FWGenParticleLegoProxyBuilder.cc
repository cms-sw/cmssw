#include "TEvePointSet.h"

#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWProxyBuilderConfiguration.h"
#include "Fireworks/Candidates/interface/CandidateUtils.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

class FWGenParticleLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::GenParticle> {

public:
   FWGenParticleLegoProxyBuilder() {}
   virtual ~FWGenParticleLegoProxyBuilder() {}

  virtual void setItem(const FWEventItem* iItem)
  {
    FWProxyBuilderBase::setItem(iItem);
    if (iItem)
      {
	iItem->getConfig()->assertParam("MarkerStyle",  0l, -1l,  3l);
	iItem->getConfig()->assertParam("MarkerSize",2., 0.1, 10.);
      }
  }
   
   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWGenParticleLegoProxyBuilder(const FWGenParticleLegoProxyBuilder&); // stop default

   const FWGenParticleLegoProxyBuilder& operator=(const FWGenParticleLegoProxyBuilder&); // stop default
   
   void build(const reco::GenParticle& iData, unsigned int iIndex,TEveElement& oItemHolder, const FWViewContext*);
};

//______________________________________________________________________________


void
FWGenParticleLegoProxyBuilder::build(const reco::GenParticle& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*) 
  {
     long markerStyle = item()->getConfig()->value<long>("MarkerStyle");
     double markerSize = item()->getConfig()->value<double>("MarkerSize");

    
    // workaround around for TEvePointSet marker styles indices
    if (markerStyle == 0 )
      markerStyle = 3;
    else if (markerStyle == 1)
      markerStyle = 4;
    else if (markerStyle == 2)
      markerStyle = 8;
    std::cerr << std::endl;

    // scale non-pixel size marker
    if (markerStyle == 3 )
      markerSize /= 20;

    TEvePointSet* ps = new TEvePointSet();
    ps->SetMarkerStyle(markerStyle);
    ps->SetMarkerSize(markerSize);
    ps->SetNextPoint(iData.eta(), iData.phi(), 0.001);
    setupAddElement( ps, &oItemHolder );

  }

REGISTER_FWPROXYBUILDER(FWGenParticleLegoProxyBuilder, reco::GenParticle, "GenParticles", FWViewType::kLegoBit);

