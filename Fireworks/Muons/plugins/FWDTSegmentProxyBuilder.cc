// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWDTSegmentProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWDTSegmentProxyBuilder.cc,v 1.1 2010/04/16 10:29:09 yana Exp $
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
#include "Fireworks/Tracks/interface/TrackUtils.h"

#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

class FWDTSegmentProxyBuilder : public FWProxyBuilderBase
{
public:
   FWDTSegmentProxyBuilder() {}
   virtual ~FWDTSegmentProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   virtual void build(const FWEventItem* iItem, TEveElementList* product);

   FWDTSegmentProxyBuilder(const FWDTSegmentProxyBuilder&);    // stop default

   const FWDTSegmentProxyBuilder& operator=(const FWDTSegmentProxyBuilder&);    // stop default
};

void
FWDTSegmentProxyBuilder::build(const FWEventItem* iItem, TEveElementList* product)
{
   const DTRecSegment4DCollection* segments = 0;
   iItem->get(segments);

   if( 0 == segments ) {
       return;
   }
   
   unsigned int index = 0;
   for( DTRecSegment4DCollection::id_iterator chamberId = segments->id_begin(), chamberIdEnd = segments->id_end();
	chamberId != chamberIdEnd; ++chamberId, ++index )
   {
      std::stringstream s;
      s << "DT Segment " << index;
      const TGeoHMatrix* matrix = iItem->getGeom()->getMatrix( (*chamberId).rawId() );

      DTRecSegment4DCollection::range range = segments->get(*chamberId);
      for( DTRecSegment4DCollection::const_iterator segment = range.first;
           segment!=range.second; ++segment)
      {
         TEveCompound* compund = new TEveCompound( "dt compound", s.str().c_str() );
         product->AddElement(compund);
         compund->OpenCompound();

         TEveStraightLineSet* segmentSet = new TEveStraightLineSet;
	 fireworks::addSegment(*segment, matrix, *segmentSet);
         segmentSet->SetLineWidth(3);
         segmentSet->SetMainColor(iItem->defaultDisplayProperties().color());
         segmentSet->SetRnrSelf(iItem->defaultDisplayProperties().isVisible());
         segmentSet->SetRnrChildren(iItem->defaultDisplayProperties().isVisible());
         compund->AddElement( segmentSet );
      }
   }
}

REGISTER_FWPROXYBUILDER( FWDTSegmentProxyBuilder, DTRecSegment4DCollection, "DT Segments", FWViewType::kAll3DBits | FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);


