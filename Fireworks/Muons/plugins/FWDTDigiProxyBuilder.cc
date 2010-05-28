// -*- C++ -*-
//
// Package:     Muon
// Class  :     FWDTDigiProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author: mccauley
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWDTDigiProxyBuilder.cc,v 1.1.2.1 2010/03/02 15:08:37 mccauley Exp $
//

#include "TEveStraightLineSet.h"
#include "TEveCompound.h"
#include "TEveGeoNode.h"

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/DTDigi/interface/DTDigiCollection.h"

class FWDTDigiProxyBuilder : public FWProxyBuilderBase
{
public:
  FWDTDigiProxyBuilder() {}
  virtual ~FWDTDigiProxyBuilder() {}

  REGISTER_PROXYBUILDER_METHODS();

private:
  virtual void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*);
  FWDTDigiProxyBuilder(const FWDTDigiProxyBuilder&);    
  const FWDTDigiProxyBuilder& operator=(const FWDTDigiProxyBuilder&);
};

void
FWDTDigiProxyBuilder::build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*)
{
  const DTDigiCollection* digis = 0;
 
  iItem->get(digis);

  if ( ! digis ) 
  {
    fwLog(fwlog::kWarning)<<"Failed to get DTDigis"<<std::endl;
    return;
  }

  for ( DTDigiCollection::DigiRangeIterator dri = digis->begin(), driEnd = digis->end(); 
        dri != driEnd; ++dri )
  {  
    const DTLayerId& dtLayerId = (*dri).first;
    const DTDigiCollection::Range& range = (*dri).second; 

    for ( DTDigiCollection::const_iterator dit = range.first;
          dit != range.second; ++dit )        
    { 
      TEveCompound* compound = new TEveCompound("dt digi compound", "dtDigis");
      compound->OpenCompound();
      product->AddElement(compound);

      TEveStraightLineSet* digiSet = new TEveStraightLineSet();
      digiSet->SetLineWidth(3);
      compound->AddElement(digiSet);
    
      int wireNumber = (*dit).wire();
      int countsTDC = (*dit).countsTDC();

      // These are probably best left for the table view
      int layerId = dtLayerId.layer();
      int superLayerId = dtLayerId.superlayerId().superLayer();
      int sectorId = dtLayerId.superlayerId().chamberId().sector();
      int stationId = dtLayerId.superlayerId().chamberId().station();
      int wheelId = dtLayerId.superlayerId().chamberId().wheel();

      /*
        We need the x position of the wire in the DT frame to create a point 
        (xPos, 0.0, 0.0) that nust be transformed to global coordinates.
        We also need the length, width, and height of the DT from the DTLayer.
        Then, rotate the tube about the proper axis by the correct angle.
      */

    } // end of iteration over digis in range
  } // end of iteration over digi range  
}

REGISTER_FWPROXYBUILDER(FWDTDigiProxyBuilder, DTDigiCollection, "DTDigi", FWViewType::kISpyBit);


