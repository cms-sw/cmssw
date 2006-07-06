#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/HcalTowerAlgo//src/HcalHardcodeGeometryData.h"

HcalGeometry::HcalGeometry(const HcalTopology * topology) 
: theTopology(topology),
  lastReqDet_(DetId::Detector(0)), 
  lastReqSubdet_(0) 
{
}
  

HcalGeometry::~HcalGeometry() {
  std::map<DetId, const CaloCellGeometry*>::iterator i;
  for (i=cellGeometries_.begin(); i!=cellGeometries_.end(); i++)
    delete i->second;
  cellGeometries_.clear();
}


std::vector<DetId> HcalGeometry::getValidDetIds(DetId::Detector det, int subdet) const {
  if (lastReqDet_!=det || lastReqSubdet_!=subdet) {
    lastReqDet_=det;
    lastReqSubdet_=subdet;
    validIds_.clear();
  }
  if (validIds_.empty()) {
    std::map<DetId, const CaloCellGeometry*>::const_iterator i;    
    for (i=cellGeometries_.begin(); i!=cellGeometries_.end(); i++)
      if (i->first.det()==det && i->first.subdetId()==subdet) 
	 validIds_.push_back(i->first);
  }

  return validIds_;
}


const DetId HcalGeometry::getClosestCell(const GlobalPoint& r) const
{
  // Now find the closest eta_bin, eta value of a bin i is average
  // of eta[i] and eta[i-1]
  double abseta = fabs(r.eta());
  
  // figure out subdetector, giving preference to HE in HE/HF overlap region
  HcalSubdetector bc= HcalEmpty;
  if( abseta <= theHBHEEtaBounds[theTopology->lastHBRing()] )
  {
    bc = HcalBarrel;
  }
  else if( abseta <= theHBHEEtaBounds[theTopology->lastHERing()] ) 
  {
    bc = HcalEndcap;
  }
  else
  {
    bc = HcalForward;
  }

  // find eta bin
  int etabin;
  if( bc == HcalForward ) {
    for(etabin = theTopology->firstHFRing(); 
        etabin <= theTopology->lastHFRing(); ++etabin)
    {
      if(theHFEtaBounds[etabin-theTopology->firstHFRing()+1] > abseta) break;
    }
  }
  else 
  {
    for(etabin = 1;
        etabin <= theTopology->lastHERing(); ++etabin)
    {
      if(theHBHEEtaBounds[etabin] > abseta) break;
    }
  }
  
  
  //Now to do phi
  double pointphi=r.phi();
  double twopi = M_PI+M_PI;
  //put phi in correct range (0->2pi)
  if(pointphi<0.0)pointphi += twopi;
  int nphibins = theTopology->nPhiBins(etabin);
  int phibin= static_cast<int>(pointphi/twopi*nphibins)+1;
  // convert to the convention of numbering 1,3,5, in 36 phi bins
  // and 1,5,9 in 18 phi bins
  phibin = (phibin-1)*(72/nphibins) + 1;

  //Now do depth if required
  int dbin = 1;
/*
  if( depth ) {
    double pointradius=point.mag();
    double dradius=99999.;
    int depthmax = geometry->hcalDepth[bc-3];
    for(int d=1; d<=depthmax; d++) 
      {
        HcalDetId id(bc,etabin,phibin, d);
        const CaloCellGeometry * cell = getGeometry(id);
        double radius=position(properties,0.0).mag();
        if(fabs(pointradius-radius)<dradius) 
          {
            dradius=fabs(pointradius-radius);
            dbin=d;
          }
      } // loop over all depths
  }
*/

  // add a sign to the etabin
  if(r.z() < 0) etabin *= -1;
  return HcalDetId(bc, etabin, phibin, dbin);
}

