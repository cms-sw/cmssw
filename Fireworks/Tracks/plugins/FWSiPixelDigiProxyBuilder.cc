// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWSiPixelDigiProxyBuilder
//
//
// Original Author:
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: FWSiPixelDigiProxyBuilder.cc,v 1.9 2010/05/06 14:14:10 mccauley Exp $
//

#include "TEveCompound.h"
#include "TEvePointSet.h"
#include "TEveVSDStructs.h"

#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"

#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/DetId/interface/DetId.h"

class FWSiPixelDigiProxyBuilder : public FWProxyBuilderBase
{
public:
  FWSiPixelDigiProxyBuilder() {}
  virtual ~FWSiPixelDigiProxyBuilder() {}

  REGISTER_PROXYBUILDER_METHODS();

private:
  virtual void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*);
  FWSiPixelDigiProxyBuilder(const FWSiPixelDigiProxyBuilder&);    
  const FWSiPixelDigiProxyBuilder& operator=(const FWSiPixelDigiProxyBuilder&);
};

void FWSiPixelDigiProxyBuilder::build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*)
{
  product->SetMainColor(iItem->defaultDisplayProperties().color());

  const edm::DetSetVector<PixelDigi>* digis = 0;
  iItem->get(digis);

  if ( ! digis )
  {
    fwLog(fwlog::kWarning)<<"ERROR: failed get SiPixelDigis"<<std::endl;
    return;
  }
  
  std::vector<TVector3> pixelDigiPoints;

  for ( edm::DetSetVector<PixelDigi>::const_iterator it = digis->begin(), end = digis->end();
        it != end; ++it )
  {
    edm::DetSet<PixelDigi> ds = *it;
    const uint32_t& detID = ds.id;
    DetId detid(detID);              
         
    for ( edm::DetSet<PixelDigi>::const_iterator idigi = ds.data.begin(), idigiEnd = ds.data.end();
          idigi != idigiEnd; ++idigi )
    {
      TEveCompound* compound = new TEveCompound("si pixel digi compound", "siPixelDigis");
      compound->OpenCompound();
      product->AddElement(compound);

      TEvePointSet* pointSet = new TEvePointSet();
      pointSet->SetMarkerSize(2);
      pointSet->SetMarkerStyle(2);
      pointSet->SetMarkerColor(46);
      compound->AddElement(pointSet);

      //int adc = static_cast<int>((*idigi).adc());
      int row = static_cast<int>((*idigi).row());
      int column = static_cast<int>((*idigi).column());
      //int channel = static_cast<int>((*idigi).channel());
      
      // This method, although called "local" seems to transform
      // the point to global coordinates. See TrackUtils.cc
      TVector3 point;
      fireworks::localSiPixel(point, row, column, detid, iItem);
      pointSet->SetNextPoint(point.x(), point.y(), point.z());
  
    } // end of iteration over digis in range   
  } // end of iteration over the DetSetVector
}


REGISTER_FWPROXYBUILDER( FWSiPixelDigiProxyBuilder,edm::DetSetVector<PixelDigi>,"SiPixelDigi", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
