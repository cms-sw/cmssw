// -*- C++ -*-
// $Id: FWSiStripClusterProxyBuilder.cc,v 1.12 2010/06/03 13:38:32 eulisse Exp $
//

#include "TEveGeoNode.h"
#include "TEveStraightLineSet.h"

#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

class FWSiStripClusterProxyBuilder : public FWSimpleProxyBuilderTemplate<SiStripCluster>
{
public:
   FWSiStripClusterProxyBuilder() {}
   virtual ~FWSiStripClusterProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

protected:
   virtual void build(const SiStripCluster& iData, unsigned int iIndex,
                      TEveElement& oItemHolder, const FWViewContext*);
   virtual void localModelChanges(const FWModelId& iId, TEveElement* iCompound,
                                  FWViewType::EType viewType, const FWViewContext* vc);

private:
   FWSiStripClusterProxyBuilder(const FWSiStripClusterProxyBuilder&);
   const FWSiStripClusterProxyBuilder& operator=(const FWSiStripClusterProxyBuilder&);              
};

void
FWSiStripClusterProxyBuilder::build(const SiStripCluster& iData,           
                                    unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*)
{  
  DetId detid(iData.geographicalId());
 
  TEveGeoShape* shape = item()->getGeom()->getShape(detid);
  
  if (shape) 
  {
    setupAddElement(shape, &oItemHolder);
    Char_t transp = item()->defaultDisplayProperties().transparency();
    shape->SetMainTransparency(TMath::Min(100, 80 + transp / 5));
  }

  else
  {
    fwLog(fwlog::kWarning) 
      <<"ERROR: failed to get shape of SiStripCluster with detid: "
      << detid <<std::endl;
  }
  
  TEveStraightLineSet *scposition = new TEveStraightLineSet( "strip" );
  
  double bc = iData.barycenter();

  TVector3 point;
  TVector3 pointA;
  TVector3 pointB;

  fireworks::localSiStrip(point, pointA, pointB, bc, detid, item());

  scposition->AddLine(pointA.X(), pointA.Y(), pointA.Z(), 
                      pointB.X(), pointB.Y(), pointB.Z() );
  
  scposition->SetLineColor(kRed);

  setupAddElement(scposition, &oItemHolder);
}

void
FWSiStripClusterProxyBuilder::localModelChanges(const FWModelId& iId, TEveElement* iCompound,
                                                FWViewType::EType viewType, const FWViewContext* vc)
{
  const FWDisplayProperties& dp = item()->modelInfo(iId.index()).displayProperties();

  TEveElement* det = iCompound->FirstChild();
  if (det)
  {
     Char_t det_transp = TMath::Min(100, 80 + dp.transparency() / 5);
     det->SetMainTransparency(det_transp);
  }
}


REGISTER_FWPROXYBUILDER(FWSiStripClusterProxyBuilder, SiStripCluster, "SiStripCluster", FWViewType::kAll3DBits | FWViewType::kAllRPZBits);
