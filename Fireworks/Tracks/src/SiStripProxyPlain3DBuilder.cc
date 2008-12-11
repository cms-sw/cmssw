// -*- C++ -*-
// $Id: SiStripProxyPlain3DBuilder.cc,v 1.4 2008/11/26 16:19:13 chrjones Exp $
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

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

#include "Fireworks/Tracks/interface/SiStripProxyPlain3DBuilder.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/src/CmsShowMain.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "TEveGeoNode.h"
#include "Fireworks/Core/src/changeElementAndChildren.h"

void SiStripProxyPlain3DBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"SiStripCluster",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
      gEve->AddElement(tList);
   } else {
      tList->DestroyElements();
   }

   const edmNew::DetSetVector<SiStripCluster>* clusters=0;
   iItem->get(clusters);

   if(0 == clusters ) return;
   std::set<DetId> modules;
   int index=0;
   for(edmNew::DetSetVector<SiStripCluster>::const_iterator set = clusters->begin();
      set != clusters->end(); ++set,++index) {
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
	    shape->SetMainTransparency(75);
	    shape->SetMainColor( iItem->defaultDisplayProperties().color() );
	    shape->SetPickable(true);
	    list->AddElement(shape);
	 }
      }

      gEve->AddElement(list,tList);
   }
}

void
SiStripProxyPlain3DBuilder::modelChanges(const FWModelIds& iIds, TEveElement* iElements)
{
   applyChangesToAllModels(iElements);
}

void
SiStripProxyPlain3DBuilder::applyChangesToAllModels(TEveElement* iElements)
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

REGISTER_FW3DDATAPROXYBUILDER(SiStripProxyPlain3DBuilder,edmNew::DetSetVector<SiStripCluster>,"SiStrip");
