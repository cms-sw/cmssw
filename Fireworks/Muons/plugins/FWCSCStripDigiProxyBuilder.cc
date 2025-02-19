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
// $Id: FWCSCStripDigiProxyBuilder.cc,v 1.14 2010/09/07 15:46:48 yana Exp $
//

#include "TEveStraightLineSet.h"
#include "TEveCompound.h"

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
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

   if( ! digis ) 
   {
      fwLog( fwlog::kWarning ) << "failed to get CSCStripDigis"<<std::endl;
      return;
   }
   const FWGeometry *geom = iItem->getGeom();

   int thresholdOffset = 9;       

   for ( CSCStripDigiCollection::DigiRangeIterator dri = digis->begin(), driEnd = digis->end();
         dri != driEnd; ++dri )
   {    
      unsigned int rawid = (*dri).first.rawId();
      const CSCStripDigiCollection::Range& range = (*dri).second;

      if( ! geom->contains( rawid ))
      {
         fwLog( fwlog::kWarning ) << "failed to get geometry of CSC with detid: "
				  << rawid << std::endl;

	 TEveCompound* compound = createCompound();
	 setupAddElement( compound, product );

	 continue;
      }
     
      const float* shape = geom->getShapePars( rawid );
      float length = shape[4];

      const float* parameters = geom->getParameters( rawid );

      float yAxisOrientation = parameters[0];
      float centreToIntersection = parameters[1];
      float yCentre = parameters[2];
      float phiOfOneEdge = parameters[3];
      float stripOffset = parameters[4];
      float angularWidth = parameters[5];

      for( CSCStripDigiCollection::const_iterator dit = range.first;
           dit != range.second; ++dit )
      {
         std::vector<int> adcCounts = (*dit).getADCCounts();
      
         int signalThreshold = (adcCounts[0] + adcCounts[1])/2 + thresholdOffset;  
        
         TEveStraightLineSet* stripDigiSet = new TEveStraightLineSet();
         setupAddElement(stripDigiSet, product);
             
         if( std::find_if( adcCounts.begin(), adcCounts.end(), bind2nd( std::greater<int>(), signalThreshold )) != adcCounts.end()) 
         {
            stripDigiSet->SetLineWidth(3);
            int stripId = (*dit).getStrip();  

            float yOrigin = centreToIntersection-yCentre;      
            float stripAngle = phiOfOneEdge + yAxisOrientation*(stripId-(0.5-stripOffset))*angularWidth;
            float tanStripAngle = tan(stripAngle);
            //float xOfStrip = yAxisOrientation*yOrigin*tanStripAngle; this is x of strip at origin
             
            float localPointTop[3] = 
              {
                yAxisOrientation*(yOrigin+length)*tanStripAngle, length, 0.0
              };

            float localPointBottom[3] = 
              {
                yAxisOrientation*(yOrigin-length)*tanStripAngle, -length, 0.0
              };
      
            float globalPointTop[3];
            float globalPointBottom[3];
        
            geom->localToGlobal( rawid, localPointTop, globalPointTop, localPointBottom, globalPointBottom);
        
            stripDigiSet->AddLine( globalPointBottom[0], globalPointBottom[1], globalPointBottom[2],
                                   globalPointTop[0], globalPointTop[1], globalPointTop[2] );
         }
      }       
   }   
}

REGISTER_FWPROXYBUILDER(FWCSCStripDigiProxyBuilder, CSCStripDigiCollection, "CSCStripDigi", 
                        FWViewType::kAll3DBits | FWViewType::kAllRPZBits);

