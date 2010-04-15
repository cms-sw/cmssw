// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWCSCSegmentProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWCSCSegmentProxyBuilder.cc,v 1.5 2010/04/09 12:59:47 amraktad Exp $
//

// system include files
#include "TEveManager.h"
#include "TEveElement.h"
#include "TEveCompound.h"
#include "TEveStraightLineSet.h"

// user include files
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"

#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"

class FWCSCSegmentProxyBuilder : public FWProxyBuilderBase
{
public:
   FWCSCSegmentProxyBuilder() {}
   virtual ~FWCSCSegmentProxyBuilder() {}
  
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWCSCSegmentProxyBuilder(const FWCSCSegmentProxyBuilder&);    // stop default
   const FWCSCSegmentProxyBuilder& operator=(const FWCSCSegmentProxyBuilder&);    // stop default

   virtual void build( const FWEventItem* iItem, TEveElementList** product );
};

void
FWCSCSegmentProxyBuilder::build( const FWEventItem* iItem, TEveElementList** product )
{
   TEveElementList* tList = *product;

   if( 0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"cscSegments",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
      gEve->AddElement(tList);
   } else {
      tList->DestroyElements();
   }

   const CSCSegmentCollection* segments = 0;
   iItem->get(segments);

   if( 0 == segments ) {
      return;
   }
   unsigned int index = 0;
   for(  CSCSegmentCollection::id_iterator chamberId = segments->id_begin();
          chamberId != segments->id_end(); ++chamberId, ++index )
   {
      const TGeoHMatrix* matrix = iItem->getGeom()->getMatrix( (*chamberId).rawId() );
      if( !matrix ) {
         std::cout << "ERROR: failed get geometry of CSC chamber with det id: " <<
         (*chamberId).rawId() << std::endl;
         continue;
      }

      std::stringstream s;
      s << "chamber" << index;

      CSCSegmentCollection::range range = segments->get(*chamberId);
      const double segmentLength = 15;
      for( CSCSegmentCollection::const_iterator segment = range.first;
           segment!=range.second; ++segment)
      {
         TEveCompound* compund = new TEveCompound("csc compound", "cscSegments");
         compund->OpenCompound();
         tList->AddElement(compund);

         TEveStraightLineSet* segmentSet = new TEveStraightLineSet(s.str().c_str());
         segmentSet->SetLineWidth(3);
         segmentSet->SetMainColor(iItem->defaultDisplayProperties().color());
         segmentSet->SetRnrSelf(iItem->defaultDisplayProperties().isVisible());
         segmentSet->SetRnrChildren(iItem->defaultDisplayProperties().isVisible());
         compund->AddElement(segmentSet);

         Double_t localSegmentInnerPoint[3];
         Double_t localSegmentCenterPoint[3];
         Double_t localSegmentOuterPoint[3];
         Double_t globalSegmentInnerPoint[3];
         Double_t globalSegmentCenterPoint[3];
         Double_t globalSegmentOuterPoint[3];

         localSegmentOuterPoint[0] = segment->localPosition().x() + segmentLength*segment->localDirection().x();
         localSegmentOuterPoint[1] = segment->localPosition().y() + segmentLength*segment->localDirection().y();
         localSegmentOuterPoint[2] = segmentLength*segment->localDirection().z();

         localSegmentCenterPoint[0] = segment->localPosition().x();
         localSegmentCenterPoint[1] = segment->localPosition().y();
         localSegmentCenterPoint[2] = 0;

         localSegmentInnerPoint[0] = segment->localPosition().x() - segmentLength*segment->localDirection().x();
         localSegmentInnerPoint[1] = segment->localPosition().y() - segmentLength*segment->localDirection().y();
         localSegmentInnerPoint[2] = -segmentLength*segment->localDirection().z();

         matrix->LocalToMaster( localSegmentInnerPoint, globalSegmentInnerPoint );
         matrix->LocalToMaster( localSegmentCenterPoint, globalSegmentCenterPoint );
         matrix->LocalToMaster( localSegmentOuterPoint, globalSegmentOuterPoint );
         if( globalSegmentInnerPoint[1] *globalSegmentOuterPoint[1] > 0 ) {
	    segmentSet->AddLine( globalSegmentInnerPoint[0], globalSegmentInnerPoint[1], globalSegmentInnerPoint[2],
				 globalSegmentOuterPoint[0], globalSegmentOuterPoint[1], globalSegmentOuterPoint[2] );
         } else {
            if( fabs(globalSegmentInnerPoint[1]) > fabs(globalSegmentOuterPoint[1]) )
               segmentSet->AddLine( globalSegmentInnerPoint[0], globalSegmentInnerPoint[1], globalSegmentInnerPoint[2],
				    globalSegmentCenterPoint[0], globalSegmentCenterPoint[1], globalSegmentCenterPoint[2] );
            else
               segmentSet->AddLine( globalSegmentCenterPoint[0], globalSegmentCenterPoint[1], globalSegmentCenterPoint[2],
				    globalSegmentOuterPoint[0], globalSegmentOuterPoint[1], globalSegmentOuterPoint[2] );
         }
      }
   }
}

REGISTER_FWPROXYBUILDER( FWCSCSegmentProxyBuilder, CSCSegmentCollection, "CSC Segments", FWViewType::k3DBit | FWViewType::kRhoPhiBit | FWViewType::kRhoZBit );


