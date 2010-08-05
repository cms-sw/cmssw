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
// $Id: FWCSCWireDigiProxyBuilder.cc,v 1.6 2010/07/29 16:56:26 mccauley Exp $
//

#include "TEveStraightLineSet.h"
#include "TEveGeoNode.h"
#include "TEveCompound.h"
#include "TGeoArb8.h"
#include "TEvePointSet.h" // rm when done testing

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

  // NOTE: these parameters are not avalaible via a public interface
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

    std::vector<float> parameters = iItem->getGeom()->getParameters(cscDetId.rawId());

    if ( parameters.empty() )
    {
      fwLog(fwlog::kWarning)<<"Parameters empty for CSC with detid: "
                            << cscDetId.rawId() <<std::endl;
      continue;
    }
    
    assert(parameters.size() == 8);

    double wireSpacing = parameters[6];
    float wireAngle = parameters[7];
    float cosWireAngle = cos(wireAngle);

    double yOfFirstWire = getYOfFirstWire(cscDetId.station(), cscDetId.ring(), length); 
             
    for ( CSCWireDigiCollection::const_iterator dit = range.first;
          dit != range.second; ++dit )        
    { 
      TEveCompound* compound = new TEveCompound("csc wire digi compound", "cscWireDigis");
      compound->OpenCompound();
      product->AddElement(compound);

      TEveStraightLineSet* wireDigiSet = new TEveStraightLineSet();
      wireDigiSet->SetLineWidth(3);
      compound->AddElement(wireDigiSet);

      TEvePointSet* testPointSet = new TEvePointSet();
      compound->AddElement(testPointSet);

      // NOTE: Can use wire group as well as wire number? Check in validation.
      int wireGroup = (*dit).getWireGroup();

      double yOfWire = yOfFirstWire + ((wireGroup-1)*wireSpacing)/cosWireAngle;
    
      // NOTE: Still need to draw wire, but since we now have the y position of the wire,
      // its angle, and the dimensions of the layer this should be straightforward
      
      double localPointCenter[3] = 
      {
        0.0, yOfWire, 0.0
      };

      double globalPointCenter[3];
       
      matrix->LocalToMaster(localPointCenter, globalPointCenter);

      testPointSet->SetNextPoint(globalPointCenter[0], globalPointCenter[1], globalPointCenter[2]);
    }
  }
}

REGISTER_FWPROXYBUILDER(FWCSCWireDigiProxyBuilder, CSCWireDigiCollection, "CSCWireDigi", FWViewType::kAll3DBits);


