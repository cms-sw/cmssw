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
  int etaring = etaRing(bc, abseta);
  
  int phibin = phiBin(r.phi(), etaring);

  // add a sign to the etaring
  int etabin = (r.z() > 0) ? etaring : -etaring;

  //Now do depth if required
  int dbin = 1;
  double pointradius=r.mag();
  double dradius=99999.;
  HcalDetId currentId(bc, etabin, phibin, dbin);
  HcalDetId bestId;
  for(  ; currentId != HcalDetId(); theTopology->incrementDepth(currentId))
  {    
    const CaloCellGeometry * cell = getGeometry(currentId);
    assert(cell != 0);
    double radius=cell->getPosition().mag();
    if(fabs(pointradius-radius)<dradius) 
    {
      bestId = currentId;
      dradius=fabs(pointradius-radius);
    }
  }

  return bestId;
}


int HcalGeometry::etaRing(HcalSubdetector bc, double abseta) const
{
  int etaring;
  if( bc == HcalForward ) {
    for(etaring = theTopology->firstHFRing();
        etaring <= theTopology->lastHFRing(); ++etaring)
    {
      if(theHFEtaBounds[etaring-theTopology->firstHFRing()+1] > abseta) break;
    }
  }
  else
  {
    for(etaring = 1;
        etaring <= theTopology->lastHERing(); ++etaring)
    {
      if(theHBHEEtaBounds[etaring] > abseta) break;
    }
  }

  return etaring;
}


int HcalGeometry::phiBin(double phi, int etaring) const
{
  double twopi = M_PI+M_PI;
  //put phi in correct range (0->2pi)
  if(phi<0.0) phi += twopi;
  int nphibins = theTopology->nPhiBins(etaring);
  int phibin= static_cast<int>(phi/twopi*nphibins)+1;

  // rings 40 and 41 are offset wrt the other phi numbering
  //  1        1         1         2
  //  ------------------------------
  //  72       36        36        1
  if(etaring >= theTopology->firstHFQuadPhiRing())
  {
    ++phibin;
    if(phibin > nphibins) phibin -= nphibins;
  }

  // convert to the convention of numbering 1,3,5, in 36 phi bins
  // and 1,5,9 in 18 phi bins
  return (phibin-1)*(72/nphibins) + 1;
}

