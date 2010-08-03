// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWCSCSegments3DProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWCSCSegments3DProxyBuilder.cc,v 1.1 2009/05/14 Yanjun Tu Exp $
//

// system include files
#include "TEveManager.h"
#include "TGeoTube.h"
#include "TEveGeoNode.h"
#include "TEveElement.h"
#include "TEveCompound.h"
#include "TEvePointSet.h"

// user include files
#include "Fireworks/Core/interface/FW3DDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/interface/FW3DView.h"
#include "Fireworks/Core/interface/FWEveScalableStraightLineSet.h"
#include "Fireworks/Core/interface/FWEveValueScaler.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"

#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
//#include "DataFormats/MuonDetId/interface/CSCChamberId.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"

class FWCSCSegments3DProxyBuilder : public FW3DDataProxyBuilder
{

public:
   FWCSCSegments3DProxyBuilder();
   virtual ~FWCSCSegments3DProxyBuilder();

   // ---------- const member functions ---------------------
   REGISTER_PROXYBUILDER_METHODS();

   // ---------- static member functions --------------------
private:
   virtual void build(const FWEventItem* iItem,
                      TEveElementList** product);

   FWCSCSegments3DProxyBuilder(const FWCSCSegments3DProxyBuilder&);    // stop default

   const FWCSCSegments3DProxyBuilder& operator=(const FWCSCSegments3DProxyBuilder&);    // stop default

   // ---------- member data --------------------------------
};

//
// constructors and destructor
//
FWCSCSegments3DProxyBuilder::FWCSCSegments3DProxyBuilder()
{
}

FWCSCSegments3DProxyBuilder::~FWCSCSegments3DProxyBuilder()
{
}

void
FWCSCSegments3DProxyBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"cscSegments",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
      gEve->AddElement(tList);
   } else {
      tList->DestroyElements();
   }

   const CSCSegmentCollection* segments = 0;
   iItem->get(segments);

   if(0 == segments ) {
      // std::cout <<"failed to get CSC segments"<<std::endl;
      return;
   }
   TEveCompound* compund = new TEveCompound("csc compound", "cscSegments" );
   compund->OpenCompound();
   unsigned int index = 0;
   for (  CSCSegmentCollection::id_iterator chamberId = segments->id_begin();
          chamberId != segments->id_end(); ++chamberId, ++index )
   {
      const TGeoHMatrix* matrix = iItem->getGeom()->getMatrix( (*chamberId).rawId() );
      if ( !matrix ) {
         std::cout << "ERROR: failed get geometry of CSC chamber with det id: " <<
         (*chamberId).rawId() << std::endl;
         continue;
      }

      std::stringstream s;
      s << "chamber" << index;
      TEveStraightLineSet* segmentSet = new TEveStraightLineSet(s.str().c_str());
      TEvePointSet* pointSet = new TEvePointSet();
      segmentSet->SetLineWidth(3);
      segmentSet->SetMainColor(iItem->defaultDisplayProperties().color());
      segmentSet->SetRnrSelf(iItem->defaultDisplayProperties().isVisible());
      segmentSet->SetRnrChildren(iItem->defaultDisplayProperties().isVisible());
      pointSet->SetMainColor(iItem->defaultDisplayProperties().color());
      compund->AddElement( segmentSet);
      segmentSet->AddElement( pointSet );

      CSCSegmentCollection::range range = segments->get(*chamberId);
      const double segmentLength = 15;
      for (CSCSegmentCollection::const_iterator segment = range.first;
           segment!=range.second; ++segment)
      {
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
         if ( globalSegmentInnerPoint[1] *globalSegmentOuterPoint[1] > 0 ) {
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
   tList->AddElement(compund);
}

REGISTER_FW3DDATAPROXYBUILDER(FWCSCSegments3DProxyBuilder,CSCSegmentCollection,"CSC-segments");


