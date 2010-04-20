// -*- C++ -*-
//
// Package:     Electrons
// Class  :     FWElectronProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  2 14:17:03 EST 2008
// $Id: FWElectronProxyBuilder.cc,v 1.9 2010/04/19 08:20:15 yana Exp $
//
#include "TEveCompound.h"
#include "TEveTrack.h"

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWViewType.h"

#include "Fireworks/Candidates/interface/CandidateUtils.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "Fireworks/Electrons/interface/makeSuperCluster.h" 

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"


////////////////////////////////////////////////////////////////////////////////
//
//   3D and RPZ proxy builder with shared track list
// 
////////////////////////////////////////////////////////////////////////////////

class FWElectronProxyBuilder : public FWProxyBuilderBase {

public:
   FWElectronProxyBuilder() ;
   virtual ~FWElectronProxyBuilder();

   virtual bool haveSingleProduct() const { return false; }
   virtual void cleanLocal();

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWElectronProxyBuilder( const FWElectronProxyBuilder& ); // stop default
   const FWElectronProxyBuilder& operator=( const FWElectronProxyBuilder& ); // stop default

   virtual void buildViewType( const FWEventItem* iItem, TEveElementList* product, FWViewType::EType type );
   TEveElementList* requestCommon( const reco::GsfElectronCollection* gsfElectrons );

   TEveElementList* m_common;
};


FWElectronProxyBuilder::FWElectronProxyBuilder():
   m_common(0)
{
   m_common = new TEveElementList( "common electron scene" );
   m_common->IncDenyDestroy();
}

FWElectronProxyBuilder::~FWElectronProxyBuilder()
{
   m_common->DecDenyDestroy();
}

TEveElementList*
FWElectronProxyBuilder::requestCommon( const reco::GsfElectronCollection* gsfElectrons )
{
   if( m_common->HasChildren() == false && gsfElectrons->empty() == false )
   {
      for( reco::GsfElectronCollection::const_iterator it = gsfElectrons->begin(), itEnd = gsfElectrons->end(); it != itEnd; ++it ) 
      {
         TEveTrack* track(0);
         if( (*it).gsfTrack().isAvailable() )
            track = fireworks::prepareTrack( *((*it).gsfTrack()),
                                             context().getTrackPropagator());
         else
            track = fireworks::prepareCandidate( (*it),
                                                 context().getTrackPropagator());
         track->MakeTrack();
         setupElement(track);
         m_common->AddElement( track );
      }
   }
   return m_common;
}

void
FWElectronProxyBuilder::cleanLocal()
{
   m_common->DestroyElements();
}

void
FWElectronProxyBuilder::buildViewType( const FWEventItem* iItem, TEveElementList* product, FWViewType::EType type )
{
   reco::GsfElectronCollection const * gsfElectrons = 0;
   iItem->get( gsfElectrons );
   if( gsfElectrons == 0 ) return;

   TEveElementList*   tracks = requestCommon( gsfElectrons );
   TEveElement::List_i trkIt = tracks->BeginChildren();

   for( reco::GsfElectronCollection::const_iterator elIt = gsfElectrons->begin(), elItEnd = gsfElectrons->end(); elIt != elItEnd; ++elIt, ++trkIt )
   { 
      TEveCompound* comp = createCompound();
      comp->AddElement( *trkIt );
      if( type == FWViewType::kRhoPhi )
         fireworks::makeRhoPhiSuperCluster( this,
					    (*elIt).superCluster(),
					    (*elIt).phi(),
					    *comp );
      else if( type == FWViewType::kRhoZ )
         fireworks::makeRhoZSuperCluster( this,
					  (*elIt).superCluster(),
					  (*elIt).phi(),
					  *comp );

      setupAddElement(comp, product);
   }
}

REGISTER_FWPROXYBUILDER( FWElectronProxyBuilder, reco::GsfElectronCollection, "Electrons", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );



////////////////////////////////////////////////////////////////////////////////
//
//   GLIMPSE specific proxy builder
// 
////////////////////////////////////////////////////////////////////////////////

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEveScalableStraightLineSet.h"

class FWElectronGlimpseProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::GsfElectron> {
public:
   FWElectronGlimpseProxyBuilder() {}
   virtual ~FWElectronGlimpseProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWElectronGlimpseProxyBuilder(const FWElectronGlimpseProxyBuilder&); // stop default

   const FWElectronGlimpseProxyBuilder& operator=(const FWElectronGlimpseProxyBuilder&); // stop default

   virtual void build(const reco::GsfElectron& iData, unsigned int iIndex, TEveElement& oItemHolder);
};

void
FWElectronGlimpseProxyBuilder::build( const reco::GsfElectron& iData, unsigned int iIndex,TEveElement& oItemHolder ) 
{
   FWEveScalableStraightLineSet* marker = new FWEveScalableStraightLineSet("", "");
   marker->SetLineWidth(2);
   fireworks::addStraightLineSegment( marker, &iData, 1.0 );
   setupAddElement(marker, &oItemHolder);
   //add to scaler at end so that it can scale the line after all ends have been added
   // FIXME: It's not a part of a standard FWSimpleProxyBuilderTemplate: the scaler is not set!
//    assert(scaler());
//    scaler()->addElement(marker);
}
REGISTER_FWPROXYBUILDER(FWElectronGlimpseProxyBuilder, reco::GsfElectron, "Electrons", FWViewType::kGlimpseBit);
