#ifndef _FWPFPATJETLEGOPROXYBUILDER_H_
#define _FWPFPATJETLEGOPROXYBUILDER_H_

// -*- C++ -*-
//
// Package:     ParticleFlow
// Class  :     FWPFPatJetLegoProxyBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Simon Harris
//

// User include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "Fireworks/ParticleFlow/interface/setTrackTypePF.h"


//-----------------------------------------------------------------------------
// FWPFPatJetLegoProxyBuilder
//-----------------------------------------------------------------------------
template <class T>
class FWPFPatJetLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<T>
{
   public:
   // ---------------- Constructor(s)/Destructor ----------------------
      FWPFPatJetLegoProxyBuilder();
      virtual ~FWPFPatJetLegoProxyBuilder();

   // --------------------- Member Functions --------------------------
      using FWProxyBuilderBase::havePerViewProduct;
      virtual bool havePerViewProduct(FWViewType::EType) const { return true; }

      using FWProxyBuilderBase::scaleProduct;
      virtual void scaleProduct(TEveElementList* parent, FWViewType::EType, const FWViewContext* vc);

      using FWProxyBuilderBase::localModelChanges;
      virtual void localModelChanges(const FWModelId& iId, TEveElement* iCompound, FWViewType::EType viewType, const FWViewContext* vc);

      using FWSimpleProxyBuilderTemplate<T>::build;
      void build( const T&, unsigned int, TEveElement&, const FWViewContext* );
   private:
      FWPFPatJetLegoProxyBuilder(const FWPFPatJetLegoProxyBuilder&);             //stop default
      const FWPFPatJetLegoProxyBuilder& operator=(FWPFPatJetLegoProxyBuilder&);  //stop default

   // --------------------- Member Functions --------------------------

};
#endif
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
