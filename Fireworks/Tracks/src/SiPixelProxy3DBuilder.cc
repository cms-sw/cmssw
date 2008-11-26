// -*- C++ -*-
//
// Package:     Core
// Class  :     SiPixelProxy3DBuilder
//
/**\class SiPixelProxy3DBuilder SiPixelProxy3DBuilder.h Fireworks/Core/interface/SiPixelProxy3DBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: SiPixelProxy3DBuilder.cc,v 1.3 2008/11/06 22:05:30 amraktad Exp $
//

// system include files
#include "TEveManager.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "RVersion.h"
#include "TEveCompound.h"
#include "TEvePointSet.h"
// #include <sstream>

// user include files
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#include "Fireworks/Tracks/interface/SiPixelProxy3DBuilder.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/src/CmsShowMain.h"
#include "TEveGeoNode.h"
#include "Fireworks/Core/src/changeElementAndChildren.h"

void SiPixelProxy3DBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"SiPixelCluster",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
      gEve->AddElement(tList);
   } else {
      tList->DestroyElements();
   }

   const SiPixelClusterCollectionNew* pixels=0;
   iItem->get(pixels);

   if(0 == pixels ) return;
   int index(0);
   for(SiPixelClusterCollectionNew::const_iterator set = pixels->begin();
       set != pixels->end(); ++set, ++index) {
      const unsigned int bufSize = 1024;
      char title[bufSize];
      char name[bufSize];
      unsigned int id = set->detId();
      snprintf(name,  bufSize,"module%d",index);
      snprintf(title, bufSize,"Module %d",id);
      TEveCompound* list = new TEveCompound(name, title);
      list->OpenCompound();
      //guarantees that CloseCompound will be called no matter what happens
      boost::shared_ptr<TEveCompound> sentry(list,boost::mem_fn(&TEveCompound::CloseCompound));
      list->SetRnrSelf(     iItem->defaultDisplayProperties().isVisible() );
      list->SetRnrChildren( iItem->defaultDisplayProperties().isVisible() );

      if (iItem->getGeom()) {
	 // const TGeoHMatrix* matrix = iItem->getGeom()->getMatrix( id );
	 TEveGeoShape* shape = iItem->getGeom()->getShape( id );
	 if(0!=shape) {
	    shape->SetMainTransparency(50);
	    shape->SetMainColor( iItem->defaultDisplayProperties().color() );
	    shape->SetPickable(true);
	    list->AddElement(shape);
	 }
      }

      gEve->AddElement(list,tList);
    }
}

void
SiPixelProxy3DBuilder::modelChanges(const FWModelIds& iIds, TEveElement* iElements)
{
   applyChangesToAllModels(iElements);
}

void
SiPixelProxy3DBuilder::applyChangesToAllModels(TEveElement* iElements)
{
   if(0!=iElements && item() && item()->size()) {
      //make the bad assumption that everything is being changed indentically
      const FWEventItem::ModelInfo info(item()->defaultDisplayProperties(),false);
      changeElementAndChildren(iElements, info);
      iElements->SetRnrSelf(info.displayProperties().isVisible());
      iElements->SetRnrChildren(info.displayProperties().isVisible());
      iElements->ElementChanged();
   }
}

REGISTER_FWRPZDATAPROXYBUILDERBASE(SiPixelProxy3DBuilder,SiPixelClusterCollectionNew,"SiPixel");
