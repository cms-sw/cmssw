// -*- C++ -*-
//
// Package:     Electrons
// Class  :     FWElectronGlimpseProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  2 14:17:03 EST 2008
// $Id: FWElectronGlimpseProxyBuilder.cc,v 1.3 2009/01/23 21:35:46 amraktad Exp $
//

// system include files
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
//#include "Fireworks/Core/interface/FWGlimpseSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEveScalableStraightLineSet.h"
#include "Fireworks/Core/interface/FWEveValueScaler.h"
#include "Fireworks/Candidates/interface/CandidateUtils.h"

class FWElectronGlimpseProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::GsfElectron> {

public:
   FWElectronGlimpseProxyBuilder()
     : m_scaler(0)
     {}
  
   virtual ~FWElectronGlimpseProxyBuilder() {}

   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWElectronGlimpseProxyBuilder(const FWElectronGlimpseProxyBuilder&); // stop default

   const FWElectronGlimpseProxyBuilder& operator=(const FWElectronGlimpseProxyBuilder&); // stop default

   virtual void build(const reco::GsfElectron& iData, unsigned int iIndex,TEveElement& oItemHolder) const;

   // FIXME: It's not a part of a standard FWSimpleProxyBuilderTemplate:
   void setScaler(FWEveValueScaler* iScaler) {
      m_scaler = iScaler;
   }
   // FIXME: It's not a part of a standard FWSimpleProxyBuilderTemplate:
   FWEveValueScaler* scaler() const {
      return m_scaler;
   }

   FWEveValueScaler* m_scaler;
};

//
// member functions
//
void
FWElectronGlimpseProxyBuilder::build(const reco::GsfElectron& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
   FWEveScalableStraightLineSet* marker = new FWEveScalableStraightLineSet("", "");
   marker->SetLineWidth(2);
   fireworks::addStraightLineSegment( marker, &iData, 1.0 );
   oItemHolder.AddElement(marker);
   //add to scaler at end so that it can scale the line after all ends have been added
   // FIXME: It's not a part of a standard FWSimpleProxyBuilderTemplate: the scaler is not set!
   assert(scaler());
   scaler()->addElement(marker);
}

REGISTER_FWPROXYBUILDER(FWElectronGlimpseProxyBuilder, std::vector<reco::GsfElectron>, "Electrons", FWViewType::kGlimpse);
