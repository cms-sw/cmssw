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
// $Id: FWCSCStripDigiProxyBuilder.cc,v 1.1.2.3 2010/06/16 12:54:11 mccauley Exp $
//

#include "TEveStraightLineSet.h"
#include "TEvePointSet.h"

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

  void testParams(int station, int ring, double* params); // this is a temp. test method
};

// This is for testing and should be moved to CSCUtils

void 
FWCSCStripDigiProxyBuilder::testParams(const int station, const int ring, double* params)
{
  if ( station == 1 )
  {
    if ( ring == 1 )
    {
      params[0] = 2.96;
          
      return;
    }
      
    if ( ring == 2 )
    { 
      params[0] = 2.33;
      
      return;
    }
      
    if ( ring == 3 )
    {
      params[0] = 2.15;

      return;
    }
      
    if ( ring == 4 )
    {
      params[0] = 2.96;
    
      return;
    }
      
    else 
      return;
  }
    
  if ( station == 2 )
  {
    if ( ring == 1 )
    {
      params[0] = 4.65;
      
      return;
    }
    
    if ( ring == 2 )
    {
      params[0] = 2.33;
    
      return;
    }
      
    else 
      return;
  }
    
  if ( station == 3 )
  {
    if ( ring == 1 )
    {
      params[0] = 4.65;
      
      return;
    }
      
    if ( ring == 2 )
    {
      params[0] = 2.33;
      
      return;
    }
      
    else 
      return;
  }
    
  if ( station == 4 )
  {
    if ( ring == 1 )
    {
      params[0] = 4.65;
     
      return;
    }
      
    if ( ring == 2 )
    {
      params[0] = 2.33;
      
      return;
    }
      
    else 
      return;
  }
    
  else
    return;
}


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

        TEvePointSet* testPointSet = new TEvePointSet();
        compound->AddElement(testPointSet);

        int stripId = (*dit).getStrip();  

        int station = cscDetId.station();
        int ring    = cscDetId.ring();

        double params[1];
        testParams(station, ring, params);

        double angularWidth = params[0]/1000.0;
        double length = 150.0; // magic number for testing

        double stripAngle = stripId*angularWidth;
        double xOfStrip = tan(stripAngle - 0.5);

        double localPointTop[3] =
          {
            xOfStrip, length*0.5, 0.0
          };

        double localPointCenter[3] = 
          {
            xOfStrip, 0.0, 0.0
          };

        double localPointBottom[3] = 
          {
            xOfStrip, -length*0.5, 0.0
          };
      
        double globalPointTop[3];
        double globalPointCenter[3];
        double globalPointBottom[3];

        matrix->LocalToMaster(localPointTop, globalPointTop);
        matrix->LocalToMaster(localPointCenter, globalPointCenter);
        matrix->LocalToMaster(localPointBottom, globalPointBottom);

        /*
        std::cout<<"CSC strip digi: "
                 << globalPointCenter[0] <<" "<< globalPointCenter[1] <<" "<< globalPointCenter[2] 
                 <<std::endl;
        */
  
        stripDigiSet->AddLine(globalPointTop[0],  globalPointTop[1],  globalPointTop[2],
                              globalPointBottom[0], globalPointBottom[1], globalPointBottom[2]);

      } 
    }       
  }   
}

REGISTER_FWPROXYBUILDER(FWCSCStripDigiProxyBuilder, CSCStripDigiCollection, "CSCStripDigi", FWViewType::kAll3DBits);

