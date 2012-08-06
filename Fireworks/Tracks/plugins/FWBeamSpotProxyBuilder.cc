/*
 *  FWBeamSpotProxyBuilder.cc
 *  FWorks
 *
 *  Created by Ianna Osborne on 7/29/10.
 *
 */
#include "TEveStraightLineSet.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

class FWBeamSpotProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::BeamSpot>
{
public:
  FWBeamSpotProxyBuilder( void ) {}
  virtual ~FWBeamSpotProxyBuilder( void ) {}
   
  REGISTER_PROXYBUILDER_METHODS();

private:
  // Disable default copy constructor
  FWBeamSpotProxyBuilder( const FWBeamSpotProxyBuilder& );
  // Disable default assignment operator
  const FWBeamSpotProxyBuilder& operator=( const FWBeamSpotProxyBuilder& );
  virtual void localModelChanges(const FWModelId& iId, TEveElement* parent, FWViewType::EType viewType, const FWViewContext* vc);

  virtual void build( const reco::BeamSpot& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* );
};

void FWBeamSpotProxyBuilder::localModelChanges(const FWModelId& iId, TEveElement* parent, FWViewType::EType viewType, const FWViewContext* vc)
{
  if( TEveStraightLineSet *ls = dynamic_cast<TEveStraightLineSet*> ( *parent->BeginChildren() ))
  { 
    Color_t c = FWProxyBuilderBase::item()->modelInfo( iId.index() ).displayProperties().color();
    for (TEveProjectable::ProjList_i j = ls->BeginProjecteds(); j != ls->EndProjecteds(); ++j)
    {
      if( TEveStraightLineSet *pls = dynamic_cast<TEveStraightLineSet*> (*j))
      {
	pls->SetMarkerColor(c);
	pls->ElementChanged();
      }
    }

    ls->SetMarkerColor(c);
    ls->ElementChanged();
  }
}

void
FWBeamSpotProxyBuilder::build( const reco::BeamSpot& bs, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* )
{  
   TEveStraightLineSet* ls = new TEveStraightLineSet();

   double pos[3] = { bs.x0(), bs.y0(), bs.z0() };
   double e[3] = { bs.x0Error(), bs.y0Error(), bs.z0Error() };

   const Int_t   N = 32;
   const Float_t S = 2*TMath::Pi()/N;

   Float_t a = e[0], b = e[1];
   for (Int_t i = 0; i<N; i++)
      ls->AddLine(a*TMath::Cos(i*S)  , b*TMath::Sin(i*S)  , 0,
                  a*TMath::Cos(i*S+S), b*TMath::Sin(i*S+S), 0);

   a = e[0]; b = e[2];
   for (Int_t i = 0; i<N; i++)
      ls->AddLine(a*TMath::Cos(i*S)  , 0, b*TMath::Sin(i*S),
                  a*TMath::Cos(i*S+S), 0, b*TMath::Sin(i*S+S));

   a = e[1]; b = e[2];
   for (Int_t i = 0; i<N; i++)
      ls->AddLine(0, a*TMath::Cos(i*S)  ,  b*TMath::Sin(i*S),
                  0, a*TMath::Cos(i*S+S),  b*TMath::Sin(i*S+S));

   ls->AddLine(0,0,0,0,0,0);
   ls->AddMarker(0,0,0);
   ls->SetMarkerStyle(21);
   const FWDisplayProperties &dp = 
      FWProxyBuilderBase::item()->defaultDisplayProperties();
   ls->SetMarkerColor( dp.color() );

   ls->RefMainTrans().SetPos(pos);
   setupAddElement(ls, &oItemHolder);
}

REGISTER_FWPROXYBUILDER( FWBeamSpotProxyBuilder, reco::BeamSpot, "Beam Spot",  FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
