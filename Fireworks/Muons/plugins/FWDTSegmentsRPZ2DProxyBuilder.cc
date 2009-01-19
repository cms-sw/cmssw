// -*- C++ -*-
//
// Package:     Muons
// Class  :     DTSegmentsProxyRhoPhiZ2DBuilder
//
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: DTSegmentsProxyRhoPhiZ2DBuilder.cc,v 1.1 2009/01/06 22:05:29 amraktad Exp $
//


#include "TEveManager.h"
#include "TEveStraightLineSet.h"
#include "TEvePointSet.h"

#include "Fireworks/Core/interface/FWRPZ2DDataProxyBuilder.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/src/changeElementAndChildren.h"

#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

class FWDTSegmentsRPZ2DProxyBuilder : public FWRPZ2DDataProxyBuilder
{

   public:
      FWDTSegmentsRPZ2DProxyBuilder();
      virtual ~FWDTSegmentsRPZ2DProxyBuilder();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------
      static void build(const FWEventItem* iItem,
                        TEveElementList** product,
                        bool rhoPhiProjection);
      // ---------- member functions ---------------------------
      REGISTER_PROXYBUILDER_METHODS();

   private:
      virtual void buildRhoPhi(const FWEventItem* iItem,
                               TEveElementList** product);

      virtual void buildRhoZ(const FWEventItem* iItem,
                               TEveElementList** product);

      virtual void modelChanges(const FWModelIds& iIds,
                                TEveElement* iElements);
      virtual void applyChangesToAllModels(TEveElement* iElements);

      FWDTSegmentsRPZ2DProxyBuilder(const FWDTSegmentsRPZ2DProxyBuilder&); // stop default

      const FWDTSegmentsRPZ2DProxyBuilder& operator=(const FWDTSegmentsRPZ2DProxyBuilder&); // stop default

      // ---------- member data --------------------------------
};


FWDTSegmentsRPZ2DProxyBuilder::FWDTSegmentsRPZ2DProxyBuilder()
{
}

FWDTSegmentsRPZ2DProxyBuilder::~FWDTSegmentsRPZ2DProxyBuilder()
{
}

void FWDTSegmentsRPZ2DProxyBuilder::buildRhoPhi(const FWEventItem* iItem, TEveElementList** product)
{
   build(iItem, product, true);
}

void FWDTSegmentsRPZ2DProxyBuilder::buildRhoZ(const FWEventItem* iItem, TEveElementList** product)
{
   build(iItem, product, false);
}

void FWDTSegmentsRPZ2DProxyBuilder::build(const FWEventItem* iItem,
					    TEveElementList** product,
					    bool rhoPhiProjection)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"dtSegments",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
      gEve->AddElement(tList);
   } else {
      tList->DestroyElements();
   }

   const DTRecSegment4DCollection* segments = 0;
   iItem->get(segments);

   if(0 == segments ) {
      // std::cout <<"failed to get DT segments"<<std::endl;
      return;
   }
   unsigned int index = 0;
   for (  DTRecSegment4DCollection::id_iterator chamberId = segments->id_begin();
	 chamberId != segments->id_end(); ++chamberId, ++index )
     {

	std::stringstream s;
	s << "chamber" << index;
	TEveStraightLineSet* segmentSet = new TEveStraightLineSet(s.str().c_str());
	TEvePointSet* pointSet = new TEvePointSet();
	pointSet->SetMarkerStyle(2);
	pointSet->SetMarkerSize(3);
	segmentSet->SetLineWidth(3);
	segmentSet->SetMainColor(iItem->defaultDisplayProperties().color());
        segmentSet->SetRnrSelf(iItem->defaultDisplayProperties().isVisible());
        segmentSet->SetRnrChildren(iItem->defaultDisplayProperties().isVisible());
	pointSet->SetMainColor(iItem->defaultDisplayProperties().color());
        gEve->AddElement( segmentSet, tList );
        segmentSet->AddElement( pointSet );
	const TGeoHMatrix* matrix = iItem->getGeom()->getMatrix( (*chamberId).rawId() );

	DTRecSegment4DCollection::range  range = segments->get(*chamberId);
	const double segmentLength = 15;
	for (DTRecSegment4DCollection::const_iterator segment = range.first;
	     segment!=range.second; ++segment)
	  {
	     Double_t localSegmentInnerPoint[3];
	     Double_t localSegmentCenterPoint[3];
	     Double_t localSegmentOuterPoint[3];
	     Double_t globalSegmentInnerPoint[3];
	     Double_t globalSegmentCenterPoint[3];
	     Double_t globalSegmentOuterPoint[3];

	     if ( (rhoPhiProjection && ! segment->hasPhi() ) ||
		  (! rhoPhiProjection && ! segment->hasZed() ) ) {
		localSegmentInnerPoint[0] = 0;
		localSegmentInnerPoint[1] = 0;
		localSegmentInnerPoint[2] = 0;
		matrix->LocalToMaster( localSegmentInnerPoint, globalSegmentInnerPoint );
		pointSet->SetNextPoint( globalSegmentInnerPoint[0], globalSegmentInnerPoint[1], globalSegmentInnerPoint[2] );
		continue;
	     }

	     localSegmentOuterPoint[0] = segment->localPosition().x() + segmentLength*segment->localDirection().x();
	     localSegmentOuterPoint[1] = segment->localPosition().y() + segmentLength*segment->localDirection().y();
	     localSegmentOuterPoint[2] = segmentLength*segment->localDirection().z();

	     localSegmentCenterPoint[0] = segment->localPosition().x();
	     localSegmentCenterPoint[1] = segment->localPosition().y();
	     localSegmentCenterPoint[2] = 0;

	     localSegmentInnerPoint[0] = segment->localPosition().x() - segmentLength*segment->localDirection().x();
	     localSegmentInnerPoint[1] = segment->localPosition().y() - segmentLength*segment->localDirection().y();
	     localSegmentInnerPoint[2] = - segmentLength*segment->localDirection().z();

	     matrix->LocalToMaster( localSegmentInnerPoint, globalSegmentInnerPoint );
	     matrix->LocalToMaster( localSegmentCenterPoint, globalSegmentCenterPoint );
	     matrix->LocalToMaster( localSegmentOuterPoint, globalSegmentOuterPoint );

	     if ( globalSegmentInnerPoint[1] * globalSegmentOuterPoint[1] > 0 ) {
		segmentSet->AddLine(globalSegmentInnerPoint[0], globalSegmentInnerPoint[1], globalSegmentInnerPoint[2],
				    globalSegmentOuterPoint[0], globalSegmentOuterPoint[1], globalSegmentOuterPoint[2] );
	     } else {
		if ( fabs(globalSegmentInnerPoint[1]) > fabs(globalSegmentOuterPoint[1]) )
		  segmentSet->AddLine(globalSegmentInnerPoint[0], globalSegmentInnerPoint[1], globalSegmentInnerPoint[2],
				      globalSegmentCenterPoint[0], globalSegmentCenterPoint[1], globalSegmentCenterPoint[2] );
		else
		  segmentSet->AddLine(globalSegmentCenterPoint[0], globalSegmentCenterPoint[1], globalSegmentCenterPoint[2],
				      globalSegmentOuterPoint[0], globalSegmentOuterPoint[1], globalSegmentOuterPoint[2] );
	     }
	  }
     }
}

void
FWDTSegmentsRPZ2DProxyBuilder::modelChanges(const FWModelIds& iIds, TEveElement* iElements)
{
   //NOTE: don't use ids() since they may not have been filled in in the build* calls
   //  since this code and FWEventItem do not agree on the # of models made

   //for now, only if all items selected will will apply the action
   //if(iIds.size() && iIds.size() == iIds.begin()->item()->size()) {
      applyChangesToAllModels(iElements);
   //}
}

void
FWDTSegmentsRPZ2DProxyBuilder::applyChangesToAllModels(TEveElement* iElements)
{
   //NOTE: don't use ids() since they may not have been filled in in the build* calls
   //  since this code and FWEventItem do not agree on the # of models made
   //if(ids().size() != 0 ) {
   if(0!=iElements && item() && item()->size()) {
      //make the bad assumption that everything is being changed indentically
      const FWEventItem::ModelInfo info(item()->defaultDisplayProperties(),false);
      changeElementAndChildren(iElements, info);
      iElements->SetRnrSelf(info.displayProperties().isVisible());
      iElements->SetRnrChildren(info.displayProperties().isVisible());
      iElements->ElementChanged();
   }
}

REGISTER_FWRPZDATAPROXYBUILDERBASE(FWDTSegmentsRPZ2DProxyBuilder,DTRecSegment4DCollection,"DT-segments");
