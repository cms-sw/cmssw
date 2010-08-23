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
// $Id: FWCSCWireDigiProxyBuilder.cc,v 1.11 2010/08/17 15:21:42 mccauley Exp $
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

  // NOTE: these parameters are not available via a public interface
  // from the geometry or topology so must be hard-coded.
  double getYOfFirstWire(const int station, const int ring, const double length);
};
       
double
FWCSCWireDigiProxyBuilder::getYOfFirstWire(const int station, const int ring, const double length)                             
{
  double yAlignmentFrame;
  double alignmentPinToFirstWire;

  if ( station == 1 ) 
  {
    yAlignmentFrame = 0.0;
 
    if ( ring == 1 || ring == 4 )
      alignmentPinToFirstWire = 1.065;
    else
      alignmentPinToFirstWire = 2.85;
  }
  
  else if ( station == 4 && ring == 1 )
  {   
    alignmentPinToFirstWire = 3.04;
    yAlignmentFrame = 3.49;
  }
  
  else if ( station == 3 && ring == 1 )
  {    
    alignmentPinToFirstWire =  2.84;
    yAlignmentFrame = 3.49;
  } 
     
  else
  {
    alignmentPinToFirstWire = 2.87;
    yAlignmentFrame = 3.49;
  }
  
  return (yAlignmentFrame-length) + alignmentPinToFirstWire;
}

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
    
    const float* parameters = iItem->getGeom()->getParameters(cscDetId.rawId());

    if ( parameters == 0 )
    {
      fwLog(fwlog::kWarning)<<"Parameters empty for CSC with detid: "
                            << cscDetId.rawId() <<std::endl;
      continue;
    }
    
    double wireSpacing = parameters[6];
    float wireAngle = parameters[7];
    float cosWireAngle = cos(wireAngle);

    double yOfFirstWire = getYOfFirstWire(cscDetId.station(), cscDetId.ring(), length); 
             
    for ( CSCWireDigiCollection::const_iterator dit = range.first;
          dit != range.second; ++dit )        
    { 
      TEveStraightLineSet* wireDigiSet = new TEveStraightLineSet();
      wireDigiSet->SetLineWidth(3);
      setupAddElement(wireDigiSet, product);

      int wireGroup = (*dit).getWireGroup();

      double yOfWire = yOfFirstWire + ((wireGroup-1)*wireSpacing)/cosWireAngle;
      yOfWire += length*0.5;

      double wireLength = yOfWire*(topWidth-bottomWidth) / length;
      wireLength += bottomWidth;
     
      double localPointLeft[3] = 
      {
        -wireLength*0.5, yOfWire, 0.0
      };

      // NOTE: This is only an approximation for slanted wires.
      // Need to improve the determination of the x coordinate.
      double localPointRight[3] = 
      {
        wireLength*0.5, yOfWire + wireLength*tan(wireAngle), 0.0
      };

      double globalPointLeft[3];     
      double globalPointRight[3];

      matrix->LocalToMaster(localPointLeft, globalPointLeft); 
      matrix->LocalToMaster(localPointRight, globalPointRight);
      
      wireDigiSet->AddLine(globalPointLeft[0],  globalPointLeft[1],  globalPointLeft[2],
                           globalPointRight[0], globalPointRight[1], globalPointRight[2]);
    }
  }
}

REGISTER_FWPROXYBUILDER(FWCSCWireDigiProxyBuilder, CSCWireDigiCollection, "CSCWireDigi", 
                        FWViewType::kAll3DBits | FWViewType::kAllRPZBits);


