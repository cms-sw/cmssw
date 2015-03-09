// -*- C++ -*-
//
#include <vector>
#include "TEveGeoNode.h"
#include "TEveLine.h"
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
    FWSiStripClusterProxyBuilder( void );
    virtual ~FWSiStripClusterProxyBuilder( void );

   REGISTER_PROXYBUILDER_METHODS();

    //    virtual void cleanLocal();
   virtual void itemBeingDestroyed(const FWEventItem*);

protected:
   using FWProxyBuilderBase::build;
   virtual void build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext*) override;
   virtual void localModelChanges( const FWModelId& iId, TEveElement* iCompound,
				   FWViewType::EType viewType, const FWViewContext* vc ) override;
private:
   FWSiStripClusterProxyBuilder( const FWSiStripClusterProxyBuilder& );
   const FWSiStripClusterProxyBuilder& operator=( const FWSiStripClusterProxyBuilder& );

   TEveElementList* m_shapeList;            
};

FWSiStripClusterProxyBuilder::FWSiStripClusterProxyBuilder() : m_shapeList(0)
{

    m_shapeList = new TEveElementList("shapePool"); 
    m_shapeList->IncDenyDestroy();
}

FWSiStripClusterProxyBuilder::~FWSiStripClusterProxyBuilder()
{
    m_shapeList->DecDenyDestroy();
}

void
FWSiStripClusterProxyBuilder::itemBeingDestroyed(const FWEventItem* iItem)
{
    m_shapeList->DestroyElements();
    FWProxyBuilderBase::itemBeingDestroyed(iItem);
}

void
FWSiStripClusterProxyBuilder::build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* )
{
   const edmNew::DetSetVector<SiStripCluster>* clusters = 0;
   iItem->get( clusters );
   if( 0 == clusters ) return;
   int cntEl = 0;

   for (TEveElement::List_i ei = product->BeginChildren(); ei != product->EndChildren(); ++ei) {
       TEveElement* holder = *ei;
       if (holder->HasChildren()) {
           holder->SetRnrSelfChildren(false, false); 
           holder->RemoveElement(holder->LastChild());
       }
   }

   // check if need to create more shapes
   int sdiff =  clusters->size() - m_shapeList->NumChildren();
   for (int i = 0; i <= sdiff; ++i) 
       m_shapeList->AddElement(new TEveGeoShape("Det"));
   
   TEveElement::List_i shapeIt = m_shapeList->BeginChildren();
   for( edmNew::DetSetVector<SiStripCluster>::const_iterator set = clusters->begin(), setEnd = clusters->end();
        set != setEnd; ++set) {

      unsigned int id = set->detId();
      const FWGeometry::GeomDetInfo& info = *iItem->getGeom()->find( id );

      double array[16] = { info.matrix[0], info.matrix[3], info.matrix[6], 0.,
			   info.matrix[1], info.matrix[4], info.matrix[7], 0.,
			   info.matrix[2], info.matrix[5], info.matrix[8], 0.,
			   info.translation[0], info.translation[1], info.translation[2], 1.
      };


      // note TEveGeoShape owns shape
      TEveGeoShape* shape = (TEveGeoShape*)(*shapeIt);
      shape->SetShape(iItem->getGeom()->getShape(info));
      shape->SetTransMatrix(array);
      shape->SetRnrSelf(true);
      shapeIt++;

      for( edmNew::DetSet<SiStripCluster>::const_iterator ic = set->begin (), icEnd = set->end (); ic != icEnd; ++ic ) 
      {
          TEveCompound* itemHolder = 0;
          TEveLine* line = 0;

          if (cntEl < product->NumChildren())
          {
              TEveElement::List_i pit = product->BeginChildren();
              std::advance(pit, cntEl);
              itemHolder = (TEveCompound*)*pit;
              itemHolder->SetRnrSelfChildren(true, true);

              line = (TEveLine*) (itemHolder->FirstChild());
              setupAddElement( shape, itemHolder );
          }
          else {
              itemHolder = createCompound(); 
              setupAddElement( itemHolder, product );
              line = new TEveLine("line");
              setupAddElement( line, itemHolder );
              setupAddElement( shape, itemHolder );
          }
          shape->SetMainTransparency(75);

          // setup line pnts
          float localTop[3] = { 0.0, 0.0, 0.0 };
          float localBottom[3] = { 0.0, 0.0, 0.0 };
          fireworks::localSiStrip( (*ic).firstStrip(), localTop, localBottom, iItem->getGeom()->getParameters( id ), id );
          float globalTop[3];
          float globalBottom[3];
          iItem->getGeom()->localToGlobal( id, localTop, globalTop, localBottom, globalBottom );
          line->SetPoint(0, globalTop[0], globalTop[1], globalTop[2]);
          line->SetPoint(1, globalBottom[0], globalBottom[1], globalBottom[2]);

          cntEl++;
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
