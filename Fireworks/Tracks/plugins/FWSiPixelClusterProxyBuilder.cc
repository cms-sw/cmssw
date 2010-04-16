// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWSiPixelClusterProxyBuilder
//
//
// Original Author:
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: FWSiPixelClusterProxyBuilder.cc,v 1.3 2010/04/16 10:59:52 amraktad Exp $
//

// system include files
#include "TEveManager.h"
#include "TEveCompound.h"
#include "TEveGeoNode.h"
#include "TEveStraightLineSet.h"

// user include files
#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
// FIXME: If it's in src, it is private and should not be used...
#include "Fireworks/Core/src/changeElementAndChildren.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "DataFormats/DetId/interface/DetId.h"

class FWSiPixelClusterProxyBuilder : public FWProxyBuilderBase
{
public:
   FWSiPixelClusterProxyBuilder() {
   }
   virtual ~FWSiPixelClusterProxyBuilder() {
   }

   REGISTER_PROXYBUILDER_METHODS();
private:
   virtual void build(const FWEventItem* iItem, TEveElementList* product);
   FWSiPixelClusterProxyBuilder(const FWSiPixelClusterProxyBuilder&);    // stop default
   const FWSiPixelClusterProxyBuilder& operator=(const FWSiPixelClusterProxyBuilder&);    // stop default
   void modelChanges(const FWModelIds& iIds, TEveElement* iElements);
   void applyChangesToAllModels(TEveElement* iElements);

protected:
   enum Mode { Clusters, Modules };
   virtual Mode getMode() { return Clusters; }
};

//______________________________________________________________________________

void FWSiPixelClusterProxyBuilder::build(const FWEventItem* iItem, TEveElementList* product)
{
   product->SetMainColor( iItem->defaultDisplayProperties().color());

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
      DetId detid(id);
      snprintf(name,  bufSize,"module%d",index);
      snprintf(title, bufSize,"Module %d",id);
      TEveCompound* list = new TEveCompound(name, title);
      list->OpenCompound();
      //guarantees that CloseCompound will be called no matter what happens
      boost::shared_ptr<TEveCompound> sentry(list,boost::mem_fn(&TEveCompound::CloseCompound));
      list->SetRnrSelf(     iItem->defaultDisplayProperties().isVisible() );
      list->SetRnrChildren( iItem->defaultDisplayProperties().isVisible() );

      if (iItem->getGeom()) {
         Mode m = getMode();
         if (m == Clusters) {
            const TGeoHMatrix* matrix = iItem->getGeom()->getMatrix( detid );
            std::vector<TVector3> pixelPoints;
            const edmNew::DetSet<SiPixelCluster> & clusters = *set;
            for (edmNew::DetSet<SiPixelCluster>::const_iterator itc = clusters.begin(), edc = clusters.end(); itc != edc; ++itc) {
               fireworks::pushPixelCluster(pixelPoints, matrix, detid, *itc);
            }
            fireworks::addTrackerHits3D(pixelPoints, list, iItem->defaultDisplayProperties().color(), 1);
         } else if (m == Modules) {
            TEveGeoShape* shape = iItem->getGeom()->getShape( id );
            if(0!=shape) {
               shape->SetMainTransparency(50);
               shape->SetMainColor( iItem->defaultDisplayProperties().color() );
               shape->SetPickable(true);
               list->AddElement(shape);
            }
         }

      }

      product->AddElement(list);
      /////////////////////////////////////////////////////	   
      //LatB
      static int C2D=0;
      static int PRINT=0;
      if (C2D) {
         if (PRINT) std::cout<<"SiPixelCluster  "<<index<<", "<<title<<std::endl;
         TEveStraightLineSet *scposition = new TEveStraightLineSet(title);
         for(edmNew::DetSet<SiPixelCluster>::const_iterator ic = set->begin (); ic != set->end (); ++ic) { 
            double lx = (*ic).x();
            double ly = (*ic).y();
            TVector3 point; fireworks::localSiPixel(point, lx, ly, id, iItem);
            static const double dd = .5;
            scposition->AddLine(point.X()-dd, point.Y(), point.Z(), point.X()+dd, point.Y(), point.Z());
            scposition->AddLine(point.X(), point.Y()-dd, point.Z(), point.X(), point.Y()+dd, point.Z());
            scposition->AddLine(point.X(), point.Y(), point.Z()-dd, point.X(), point.Y(), point.Z()+dd);

            scposition->SetLineColor(kRed);
         }
      }
   }
}

void
FWSiPixelClusterProxyBuilder::modelChanges(const FWModelIds& iIds, TEveElement* iElements)
{
   applyChangesToAllModels(iElements);
}

void
FWSiPixelClusterProxyBuilder::applyChangesToAllModels(TEveElement* iElements)
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

class FWSiPixelClusterModProxyBuilder : public FWSiPixelClusterProxyBuilder {
public:
   FWSiPixelClusterModProxyBuilder() {}
    ~FWSiPixelClusterModProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();
protected:
    virtual Mode getMode() { return Modules; }
};

REGISTER_FWPROXYBUILDER(FWSiPixelClusterProxyBuilder,SiPixelClusterCollectionNew,"SiPixel", FWViewType::kAll3DBits | FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);
REGISTER_FWPROXYBUILDER(FWSiPixelClusterModProxyBuilder,SiPixelClusterCollectionNew,"SiPixelDets", FWViewType::kAll3DBits | FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);
