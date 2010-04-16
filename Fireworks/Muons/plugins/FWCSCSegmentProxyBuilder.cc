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
// $Id: FWCSCSegmentProxyBuilder.cc,v 1.1 2010/04/15 14:46:44 yana Exp $
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
      tList =  new TEveElementList(iItem->name().c_str(), "cscSegments", true);
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
   for(  CSCSegmentCollection::id_iterator chamberId = segments->id_begin(), chamberIdEnd = segments->id_end();
          chamberId != chamberIdEnd; ++chamberId, ++index )
   {
      const TGeoHMatrix* matrix = iItem->getGeom()->getMatrix( (*chamberId).rawId() );
      if( !matrix ) {
         std::cout << "ERROR: failed get geometry of CSC chamber with det id: "
		   << (*chamberId).rawId() << std::endl;
         continue;
      }

      std::stringstream s;
      s << "CSC Segment " << index;

      CSCSegmentCollection::range range = segments->get( *chamberId );
      for( CSCSegmentCollection::const_iterator segment = range.first;
           segment != range.second; ++segment)
      {
         TEveCompound* compund = new TEveCompound("csc compound", s.str().c_str() );
         compund->OpenCompound();
         tList->AddElement(compund);

         TEveStraightLineSet* segmentSet = new TEveStraightLineSet;
	 fireworks::addSegment(*segment, matrix, *segmentSet);
         segmentSet->SetLineWidth( 3 );
         segmentSet->SetMainColor( iItem->defaultDisplayProperties().color() );
         segmentSet->SetRnrSelf( iItem->defaultDisplayProperties().isVisible() );
         segmentSet->SetRnrChildren( iItem->defaultDisplayProperties().isVisible() );
         compund->AddElement( segmentSet );
      }
   }
}

REGISTER_FWPROXYBUILDER( FWCSCSegmentProxyBuilder, CSCSegmentCollection, "CSC Segments", FWViewType::k3DBit | FWViewType::kRhoPhiBit | FWViewType::kRhoZBit );


