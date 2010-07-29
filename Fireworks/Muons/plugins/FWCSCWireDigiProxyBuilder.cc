// -*- C++ -*-
//
// Package:     Muon
// Class  :     FWCSCWireDigiProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author: mccauley
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWCSCWireDigiProxyBuilder.cc,v 1.5 2010/07/28 16:46:33 mccauley Exp $
//

#include "TEveStraightLineSet.h"
#include "TEveGeoNode.h"
#include "TEveCompound.h"
#include "TGeoArb8.h"

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"

#include <cmath>

class FWCSCWireDigiProxyBuilder : public FWProxyBuilderBase
{
public:
  FWCSCWireDigiProxyBuilder() {}
  virtual ~FWCSCWireDigiProxyBuilder() {}

  REGISTER_PROXYBUILDER_METHODS();

private:
  virtual void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*);
  FWCSCWireDigiProxyBuilder(const FWCSCWireDigiProxyBuilder&);    
  const FWCSCWireDigiProxyBuilder& operator=(const FWCSCWireDigiProxyBuilder&);

  void testParams(const int station, const int ring, double* params);
};
       
void
FWCSCWireDigiProxyBuilder::build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*)
{
  const CSCWireDigiCollection* digis = 0;
 
  iItem->get(digis);

  if ( ! digis ) 
  {
    fwLog(fwlog::kWarning)<<"Failed to get CSCWireDigis"<<std::endl;
    return;
  }

  for ( CSCWireDigiCollection::DigiRangeIterator dri = digis->begin(), driEnd = digis->end(); 
        dri != driEnd; ++dri )
  { 
    const CSCDetId& cscDetId = (*dri).first;
    const CSCWireDigiCollection::Range& range = (*dri).second;
 
    const TGeoHMatrix* matrix = iItem->getGeom()->getMatrix(cscDetId.rawId());
    
    if ( ! matrix )
    {
      fwLog(fwlog::kWarning)<<"Failed to get geometry of CSC chamber with detid: "
                            << cscDetId.rawId() <<std::endl;
      continue;
    }

    TEveGeoShape* shape = iItem->getGeom()->getShape(cscDetId.rawId());

    if ( ! shape )
    {
      fwLog(fwlog::kWarning)<<"Failed to get shape of CSC chamber with detid: "
                            << cscDetId.rawId() <<std::endl;
      continue;
    }

    double length;
    double topWidth;
    double bottomWidth;
 
    if ( TGeoTrap* trap = dynamic_cast<TGeoTrap*>(shape->GetShape()) )
    {
      topWidth = trap->GetTl1()*2.0;
      bottomWidth = trap->GetBl1()*2.0;
      length = trap->GetH1()*2.0;
    }

    else
    {
      fwLog(fwlog::kWarning)<<"Failed to get trapezoid from shape for CSC with detid: "
                            << cscDetId.rawId() <<std::endl;
      continue;
    }
     
    for ( CSCWireDigiCollection::const_iterator dit = range.first;
          dit != range.second; ++dit )        
    { 
      TEveCompound* compound = new TEveCompound("csc wire digi compound", "cscWireDigis");
      compound->OpenCompound();
      product->AddElement(compound);

      TEveStraightLineSet* wireDigiSet = new TEveStraightLineSet();
      wireDigiSet->SetLineWidth(3);
      compound->AddElement(wireDigiSet);

      int wireGroup = (*dit).getWireGroup();

      int station = cscDetId.station();
      int ring    = cscDetId.ring();

      /*
        Note:

        These numbers are fetched from cscSpecs.xml
        We should think carefully about the interface when full
        framework is available.
        It seems that they DO NOT come from DDD.
      */

      if ( station == 1 && ring == 4 )
      {
        fwLog(fwlog::kWarning)<<"ME1a not handled yet"<<std::endl;
        continue;
      }

      double wireSpacing;

      if ( ring == 1 )
      {
        if ( station == 1 )
          wireSpacing = 2.5; // mm
        else 
          wireSpacing = 3.12; // mm
      }
      
      else
        wireSpacing = 3.16; // mm 
      

      double alignmentPinToFirstWire;

      if ( station == 1 && ring == 1 )
        alignmentPinToFirstWire = 10.65; // mm
      else
        alignmentPinToFirstWire = 29.0; // mm


      double yAlignmentFrame;

      if ( station == 1 && ring == 1 )
        yAlignmentFrame = 0.0; // cm
      else 
        yAlignmentFrame = 3.49; // cm

      
      double yOfFirstWire = yAlignmentFrame*10.0 + alignmentPinToFirstWire;
      
      // Wires are only ganged in ME1a? 
      // If so, then this should work with all except that chamber.

      double yOfWire = yOfFirstWire + (wireGroup-1)*wireSpacing;
      
      double lengthOfWireGroup = (topWidth - bottomWidth)*0.5 / length;
      lengthOfWireGroup *= yOfWire;
      yOfWire -= length;

      double localPointLeft[3] =
      {
        -lengthOfWireGroup*0.5, yOfWire, 0.0
      };
      
      double localPointCenter[3] = 
      {
        0.0, yOfWire, 0.0
      };

      double localPointRight[3] = 
      {
        lengthOfWireGroup*0.5, yOfWire, 0.0
      };

      double globalPointLeft[3];
      double globalPointCenter[3];
      double globalPointRight[3];

      /*
      std::cout<<"CSC wire digi: "
                 << globalPointCenter[0] <<" "<< globalPointCenter[1] <<" "<< globalPointCenter[2] 
                 <<std::endl;
      */

      matrix->LocalToMaster(localPointLeft,   globalPointLeft);
      matrix->LocalToMaster(localPointCenter, globalPointCenter);
      matrix->LocalToMaster(localPointRight,  globalPointRight);

      wireDigiSet->AddLine(globalPointLeft[0],  globalPointLeft[1],  globalPointLeft[2],
                           globalPointRight[0], globalPointRight[1], globalPointRight[2]);

    }
  }
}

REGISTER_FWPROXYBUILDER(FWCSCWireDigiProxyBuilder, CSCWireDigiCollection, "CSCWireDigi", FWViewType::kAll3DBits);


