// -*- C++ -*-
// $Id: FWSiStripClusterProxyBuilder.cc,v 1.20 2012/06/01 04:02:45 amraktad Exp $
//

#include "TEveGeoNode.h"
#include "TEveStraightLineSet.h"
#include "TEveCompound.h"

#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

class FWSiStripClusterProxyBuilder : public FWProxyBuilderBase
{
public:
   FWSiStripClusterProxyBuilder( void ) {}
   virtual ~FWSiStripClusterProxyBuilder( void ) {}

   REGISTER_PROXYBUILDER_METHODS();

   virtual void clean();

protected:
   virtual void build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext*);
   virtual void localModelChanges( const FWModelId& iId, TEveElement* iCompound,
				   FWViewType::EType viewType, const FWViewContext* vc );
private:
   FWSiStripClusterProxyBuilder( const FWSiStripClusterProxyBuilder& );
   const FWSiStripClusterProxyBuilder& operator=( const FWSiStripClusterProxyBuilder& );              
};


void
FWSiStripClusterProxyBuilder::clean()
{
   // keep itemholders to restore configuration

   for (FWProxyBuilderBase::Product_it i = m_products.begin(); i != m_products.end(); ++i)
   {
      if ((*i)->m_elements)
      {
         TEveElement* elms = (*i)->m_elements;
         for (TEveElement::List_i it = elms->BeginChildren(); it != elms->EndChildren(); ++it)
            (*it)->DestroyElements();
      }
   }

   cleanLocal();
}
void
FWSiStripClusterProxyBuilder::build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* )
{
   const edmNew::DetSetVector<SiStripCluster>* clusters = 0;
   iItem->get( clusters );
   if( 0 == clusters ) return;
   int cnt = 0;

   for( edmNew::DetSetVector<SiStripCluster>::const_iterator set = clusters->begin(), setEnd = clusters->end();
        set != setEnd; ++set) {
      unsigned int id = set->detId();

      
      TEveGeoShape* shape = item()->getGeom()->getEveShape( id );
      if (shape) 
      {
         shape->SetMainTransparency( 75 );    
         shape->SetElementName( "Det" );
      }
      else      
      {
         fwLog( fwlog::kWarning ) 
            << "failed to get shape of SiStripCluster with detid: "
            << id << std::endl;
      }  

      for( edmNew::DetSet<SiStripCluster>::const_iterator ic = set->begin (), icEnd = set->end (); ic != icEnd; ++ic ) 
      {
         TEveCompound* itemHolder = 0;
         if (cnt < product->NumChildren())
         {
            TEveElement::List_i pit = product->BeginChildren();
            std::advance(pit, cnt);
            itemHolder = (TEveCompound*)*pit;
            itemHolder->SetRnrSelfChildren(true, true);
         }
         else {
            itemHolder = createCompound(); 
            setupAddElement( itemHolder, product );
         }

         // add common shape
         if (shape) 
         {
            setupAddElement( shape, itemHolder );
            increaseComponentTransparency( cnt, itemHolder, "Det", 60 );
         }

         // add line        
         if( ! item()->getGeom()->contains( id ))
         {
            fwLog( fwlog::kError )
               << "failed to geometry of SiStripCluster with detid: " 
               << id << std::endl;
            continue;
         }

         TEveStraightLineSet *lineSet = new TEveStraightLineSet( "strip" );
         setupAddElement( lineSet, itemHolder ); 
         float localTop[3] = { 0.0, 0.0, 0.0 };
         float localBottom[3] = { 0.0, 0.0, 0.0 };

         fireworks::localSiStrip( (*ic).firstStrip(), localTop, localBottom, iItem->getGeom()->getParameters( id ), id );

         float globalTop[3];
         float globalBottom[3];
         iItem->getGeom()->localToGlobal( id, localTop, globalTop, localBottom, globalBottom );
  
         lineSet->AddLine( globalTop[0], globalTop[1], globalTop[2],
                           globalBottom[0], globalBottom[1], globalBottom[2] ); 

         cnt++;
      }
   }
}
void
FWSiStripClusterProxyBuilder::localModelChanges( const FWModelId& iId, TEveElement* iCompound,
                                                 FWViewType::EType viewType, const FWViewContext* vc )
{
  increaseComponentTransparency( iId.index(), iCompound, "Det", 60 );
}

REGISTER_FWPROXYBUILDER( FWSiStripClusterProxyBuilder, edmNew::DetSetVector<SiStripCluster>, "SiStripCluster", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
