// -*- C++ -*-
//
// Package:     Core
// Class  :     FWSiPixeRPZProxyBuilder
//
/**\class FWSiPixeRPZProxyBuilder FWSiPixeRPZProxyBuilder.h Fireworks/Core/interface/FWSiPixeRPZProxyBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: FWSiPixeRPZProxyBuilder.cc,v 1.1 2009/01/16 16:19:13 chrjones Exp $
//

// system include files
#include "TEveManager.h"
#include "TEveCompound.h"
#include "TEveGeoNode.h"

// user include files
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/src/CmsShowMain.h"
#include "Fireworks/Core/src/changeElementAndChildren.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

class FWSiPixeRPZProxyBuilder : public FWRPZDataProxyBuilder
{
   public:
      FWSiPixeRPZProxyBuilder() {}
      virtual ~FWSiPixeRPZProxyBuilder() {}
      REGISTER_PROXYBUILDER_METHODS();
   private:
      virtual void build(const FWEventItem* iItem, TEveElementList** product);

      FWSiPixeRPZProxyBuilder(const FWSiPixeRPZProxyBuilder&); // stop default

      const FWSiPixeRPZProxyBuilder& operator=(const FWSiPixeRPZProxyBuilder&); // stop default

      void modelChanges(const FWModelIds& iIds, TEveElement* iElements);
      void applyChangesToAllModels(TEveElement* iElements);
};

void FWSiPixeRPZProxyBuilder::build(const FWEventItem* iItem, TEveElementList** product)
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
FWSiPixeRPZProxyBuilder::modelChanges(const FWModelIds& iIds, TEveElement* iElements)
{
   applyChangesToAllModels(iElements);
}

void
FWSiPixeRPZProxyBuilder::applyChangesToAllModels(TEveElement* iElements)
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

REGISTER_FWRPZDATAPROXYBUILDERBASE(FWSiPixeRPZProxyBuilder,SiPixelClusterCollectionNew,"SiPixel");
