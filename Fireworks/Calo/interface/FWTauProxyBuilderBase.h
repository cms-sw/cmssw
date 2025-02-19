#ifndef Fireworks_Calo_FWTauProxyBuilderBase_h
#define Fireworks_Calo_FWTauProxyBuilderBase_h

// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWTauProxyBuilderBase
// 
/**\class FWTauProxyBuilderBase FWTauProxyBuilderBase.h Fireworks/Calo/interface/FWTauProxyBuilderBase.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel 
//         Created:  Thu Oct 21 20:40:32 CEST 2010
// $Id: FWTauProxyBuilderBase.h,v 1.2 2012/03/23 00:08:28 amraktad Exp $
//

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWViewType.h"
#include "Fireworks/Calo/interface/scaleMarker.h"
#include <vector>

class TEveScalableStraightLineSet;
class FWViewContext;

namespace reco
{
class Jet;
class BaseTau;
} 

namespace fireworks
{
class Context;
}

class FWTauProxyBuilderBase : public FWProxyBuilderBase
{
public:
   FWTauProxyBuilderBase();
   virtual ~FWTauProxyBuilderBase();

   virtual bool haveSingleProduct() const { return false; }
   virtual bool havePerViewProduct(FWViewType::EType) const { return true; }
   virtual void scaleProduct(TEveElementList* parent, FWViewType::EType, const FWViewContext* vc);
   virtual void cleanLocal();

protected:
   float m_minTheta;
   float m_maxTheta;
   std::vector<double> m_phis;
   void buildBaseTau( const reco::BaseTau& iTau, const reco::Jet* iJet, TEveElement* comp, FWViewType::EType type, const FWViewContext* vc);

   virtual void localModelChanges(const FWModelId& iId, TEveElement* iCompound,
                                  FWViewType::EType viewType, const FWViewContext* vc);
private:
   FWTauProxyBuilderBase(const FWTauProxyBuilderBase&); // stop default

   const FWTauProxyBuilderBase& operator=(const FWTauProxyBuilderBase&); // stop default

   // ---------- member data --------------------------------
   // Add Tracks which passed quality cuts and
   // are inside a tracker signal cone around leading Track
   void addConstituentTracks( const reco::BaseTau &tau, class TEveElement *product );
   // Add leading Track
   void addLeadTrack( const reco::BaseTau &tau, class TEveElement *product );
   std::vector<fireworks::scaleMarker> m_lines;
};


#endif
