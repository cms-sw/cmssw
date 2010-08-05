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
// $Id: FWCSCStripDigiProxyBuilder.cc,v 1.4 2010/07/29 15:56:23 mccauley Exp $
//

#include "TEveStraightLineSet.h"
#include "TEvePointSet.h" // rm when done testing

#include "TEveGeoNode.h"
#include "TEveCompound.h"

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"

#include <TGeoArb8.h>

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

  /*
  double width  = 0.01;
  double depth  = 0.01;
  double rotate = 0.0;
  */
  int thresholdOffset = 9;       

  for ( CSCStripDigiCollection::DigiRangeIterator dri = digis->begin(), driEnd = digis->end();
        dri != driEnd; ++dri )
  {    
    const CSCDetId& cscDetId = (*dri).first;
    const CSCStripDigiCollection::Range& range = (*dri).second;

    const TGeoHMatrix* matrix = iItem->getGeom()->getMatrix(cscDetId.rawId());
    
    if ( ! matrix )
    {
      fwLog(fwlog::kWarning)<<"Failed to get geometry of CSC with detid: "
                            << cscDetId.rawId() <<std::endl;
      continue;
    }
     
    TEveGeoShape* shape = iItem->getGeom()->getShape(cscDetId.rawId());

    if ( ! shape )
    {
      fwLog(fwlog::kWarning)<<"Failed to get shape of CSC with detid: "
                            << cscDetId.rawId() <<std::endl;
      continue;
    }

    double length;

    if ( TGeoTrap* trap = dynamic_cast<TGeoTrap*>(shape->GetShape()) )
      length = trap->GetH1()*2.0;

    else
    {
      fwLog(fwlog::kWarning)<<"Failed to get trapezoid from shape for CSC with detid: "
                            << cscDetId.rawId() <<std::endl;
      continue;
    }

    std::vector<float> parameters = iItem->getGeom()->getParameters(cscDetId.rawId());
      
    if ( parameters.empty() )
    {
      fwLog(fwlog::kWarning)<<"Parameters empty for CSC layer with detid: " 
                            << cscDetId.rawId() <<std::endl;
      continue;
    }

    assert(parameters.size() == 8);

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
      
      TEveCompound* compound = new TEveCompound("csc strip digi compound", "cscStripDigis");
      compound->OpenCompound();
      product->AddElement(compound);

      int signalThreshold = (adcCounts[0] + adcCounts[1])/2 + thresholdOffset;  
            
      if ( std::find_if(adcCounts.begin(),adcCounts.end(),bind2nd(std::greater<int>(),signalThreshold)) != adcCounts.end() ) 
      {
        TEveStraightLineSet* stripDigiSet = new TEveStraightLineSet();
        stripDigiSet->SetLineWidth(3);
        compound->AddElement(stripDigiSet);

        TEvePointSet* testPointSet = new TEvePointSet();
        compound->AddElement(testPointSet);

        int stripId = (*dit).getStrip();  

        double stripAngle = phiOfOneEdge + yAxisOrientation*(stripId-(0.5-stripOffset))*angularWidth;
        double xOfStrip = yAxisOrientation*(centreToIntersection-yCentre)*tan(stripAngle);

        double localPoint[3] = 
          {
            xOfStrip, 0.0, 0.0
          };
      
        double globalPoint[3];

        matrix->LocalToMaster(localPoint, globalPoint);
       
        testPointSet->SetNextPoint(globalPoint[0], globalPoint[1], globalPoint[2]);

      } 
    }       
  }   
}

REGISTER_FWPROXYBUILDER(FWCSCStripDigiProxyBuilder, CSCStripDigiCollection, "CSCStripDigi", FWViewType::kAll3DBits);

