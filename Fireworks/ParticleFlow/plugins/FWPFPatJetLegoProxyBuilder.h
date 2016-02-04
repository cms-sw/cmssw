#ifndef _FWPFAPATJETLEGOPROXYBUILDER_H_
#define _FWPFPATJETLEGOPROXYBUILDER_H_

// User include files

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "Fireworks/ParticleFlow/interface/FWLegoEvePFCandidate.h"

#include "Fireworks/ParticleFlow/interface/setTrackTypePF.h"


template <class T>
class FWPFPatJetLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<T> {
public:
   FWPFPatJetLegoProxyBuilder();
   virtual ~FWPFPatJetLegoProxyBuilder();

   // -------------------- member functions --------------------------
   virtual bool havePerViewProduct(FWViewType::EType) const { return true; }
   virtual void scaleProduct(TEveElementList* parent, FWViewType::EType, const FWViewContext* vc);
   virtual void localModelChanges(const FWModelId& iId, TEveElement* iCompound, FWViewType::EType viewType, const FWViewContext* vc);

private:
   FWPFPatJetLegoProxyBuilder(const FWPFPatJetLegoProxyBuilder&);             //stop default
   const FWPFPatJetLegoProxyBuilder& operator=(FWPFPatJetLegoProxyBuilder&);  //stop default

   void build(const T&, unsigned int, TEveElement&, const FWViewContext*);

};
#endif
