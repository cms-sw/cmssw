// -*- C++ -*-
//
// Package:     Muon
// Class  :     FWCSCStripDigiProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author: mccauley
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWCSCStripDigiProxyBuilder.cc,v 1.1.2.3 2010/03/16 14:51:27 mccauley Exp $
//

#include "TEveStraightLineSet.h"
#include "TEveGeoNode.h"
#include "TEveCompound.h"

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"

class FWCSCStripDigiProxyBuilder : public FWProxyBuilderBase
{
public:
  FWCSCStripDigiProxyBuilder() {}
  virtual ~FWCSCStripDigiProxyBuilder() {}

  REGISTER_PROXYBUILDER_METHODS();

private:
  virtual void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*);
  FWCSCStripDigiProxyBuilder(const FWCSCStripDigiProxyBuilder&);    
  const FWCSCStripDigiProxyBuilder& operator=(const FWCSCStripDigiProxyBuilder&);
};


void
FWCSCStripDigiProxyBuilder::build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*)
{
  const CSCStripDigiCollection* digis = 0;
  
  iItem->get(digis);

  if ( ! digis ) 
  {
     fwLog(fwlog::kWarning)<<"Failed to get CSCStripDigis"<<std::endl;
     return;
  }

  double width  = 0.01;
  double depth  = 0.01;
  double rotate = 0.0;
  int thresholdOffset = 9;       

  for ( CSCStripDigiCollection::DigiRangeIterator dri = digis->begin(), driEnd = digis->end();
        dri != driEnd; ++dri )
  {    
    const CSCDetId& cscDetId = (*dri).first;
    const CSCStripDigiCollection::Range& range = (*dri).second;

    const TGeoHMatrix* matrix = iItem->getGeom()->getMatrix(cscDetId.rawId());
    
    if ( ! matrix )
    {
      fwLog(fwlog::kWarning)<<"Failed to get geometry of CSC chamber with detid: "
                            << cscDetId.rawId() <<std::endl;
      continue;
    }
     
    for( CSCStripDigiCollection::const_iterator dit = range.first;
         dit != range.second; ++dit )
    {
      std::vector<int> adcCounts = (*dit).getADCCounts();
      
      TEveCompound* compound = new TEveCompound("csc strip digi compound", "cscStripDigis");
      compound->OpenCompound();
      product->AddElement(compound);

      int signalThreshold = (adcCounts[0] + adcCounts[1])/2 + thresholdOffset;  
            
      if ( std::find_if(adcCounts.begin(),adcCounts.end(),bind2nd(std::greater<int>(),signalThreshold)) != adcCounts.end() ) 
      {
        TEveStraightLineSet* stripDigiSet = new TEveStraightLineSet();
        stripDigiSet->SetLineWidth(3);
        compound->AddElement(stripDigiSet);

        int stripId = (*dit).getStrip();  
        int endcap  = cscDetId.endcap();
        int station = cscDetId.station();
        int ring    = cscDetId.ring();
        int chamber = cscDetId.chamber();

        /*
        std::cout<<"stripId, endcap, station, ring. chamber: "
                 << stripId <<" "<< endcap <<" "<< station <<" "
                 << ring <<" "<< chamber <<std::endl;
        */

        /*
          We need the x position of the strip to create 
          a local position: (xStrip(stripId), 0.0, 0.1)
          and then a conversion to global coordinates.
          We also need the length of the strip.
          
          The strip digi is rotated about the z axis by an angle:
        
          double angle = -atan2(pos.x(),pos.y()) - rotate;
      
          and a "box" is drawn with the width, length, and depth given above
        */
      } 
    }       
  }   
}

REGISTER_FWPROXYBUILDER(FWCSCStripDigiProxyBuilder, CSCStripDigiCollection, "CSCStripDigi", FWViewType::kISpyBit);

