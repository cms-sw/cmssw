#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/HcalTowerAlgo/interface/HcalDDDGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HcalDDDGeometry::HcalDDDGeometry() : lastReqDet_(DetId::Detector(0)), 
				     lastReqSubdet_(0), etaMax_(0),
				     firstHFQuadRing_(40) {
  twopi = M_PI + M_PI;
  deg   = M_PI/180.;
}


HcalDDDGeometry::~HcalDDDGeometry() {}


std::vector<DetId> const & HcalDDDGeometry::getValidDetIds(DetId::Detector det,
						   int subdet) const {

  if (lastReqDet_!=det || lastReqSubdet_!=subdet) {
    lastReqDet_=det;
    lastReqSubdet_=subdet;
    validIds_.clear();
  }

  if (validIds_.empty()) {
    validIds_.reserve(cellGeometries().size());
    CaloSubdetectorGeometry::CellCont::const_iterator i;
    for (i=cellGeometries().begin(); i!=cellGeometries().end(); i++)
      if (i->first.det()==det && i->first.subdetId()==subdet) 
	validIds_.push_back(i->first);
    std::sort(validIds_.begin(),validIds_.end());
  }

  LogDebug("HCalGeom") << "HcalDDDGeometry::getValidDetIds: "
		       << validIds_.size() << " valid IDs found for detector "
		       << det << " Sub-detector " << subdet;
  return validIds_;
}


const DetId HcalDDDGeometry::getClosestCell(const GlobalPoint& r) const {

  // Now find the closest eta_bin, eta value of a bin i is average
  // of eta[i] and eta[i-1]
  double abseta = fabs(r.eta());
  double phi    = r.phi();
  if (phi < 0) phi += twopi;
  double radius = r.mag();
  double z      = fabs(r.z());
  LogDebug("HCalGeom") << "HcalDDDGeometry::getClosestCell for eta "
		       << r.eta() << " phi " << phi/deg << " z " << r.z()
		       << " radius " << radius;

  HcalDetId bestId;
  if (abseta <= etaMax_) {
    for (unsigned int i=0; i<hcalCells_.size(); i++) {
      if (abseta >=hcalCells_[i].etaMin() && abseta <=hcalCells_[i].etaMax()) {
	HcalSubdetector bc = hcalCells_[i].detType();
	int etaring = hcalCells_[i].etaBin();
	int phibin  = static_cast<int>(((phi/deg)+hcalCells_[i].phiOffset())/
				       hcalCells_[i].phiBinWidth()) + 1;
	// rings 40 and 41 are offset wrt the other phi numbering
	//  1        1         1         2
	//  ------------------------------
	//  72       36        36        1
	if (etaring >= firstHFQuadRing_) {
	  ++phibin;
	  if (phibin > hcalCells_[i].nPhiBins()) 
	    phibin -= hcalCells_[i].nPhiBins();
	}
 
	// convert to the convention of numbering 1,3,5, in 36 phi bins
	// and 1,5,9 in 18 phi bins
	phibin     = (phibin-1)*(hcalCells_[i].nPhiModule()) + 1;
	int dbin   = 1;
	int etabin = (r.z() > 0) ? etaring : -etaring;
	if (bc == HcalForward) {
	  bestId   = HcalDetId(bc, etabin, phibin, dbin);
	  break;
	} else {
	  double rz = z;
	  if (hcalCells_[i].depthType()) rz = radius;
	  if (rz < hcalCells_[i].depthMax()) {
	    dbin   = hcalCells_[i].depthSegment();
	    bestId = HcalDetId(bc, etabin, phibin, dbin);
	    break;
	  }
	}
      }
    }
  }
  LogDebug("HCalGeom") << "HcalDDDGeometry::getClosestCell " << bestId;
  
  return bestId;
}


int HcalDDDGeometry::insertCell(std::vector<HcalCellType::HcalCellType> const & cells){

  hcalCells_.insert(hcalCells_.end(), cells.begin(), cells.end());
  int num = static_cast<int>(hcalCells_.size());
  for (unsigned int i=0; i<cells.size(); i++) {
    if (cells[i].etaMax() > etaMax_ ) etaMax_ = cells[i].etaMax();
  }
  LogDebug("HCalGeom") << "HcalDDDGeometry::insertCell " << cells.size()
		       << " cells inserted == Total " << num
		       << " EtaMax = " << etaMax_;
  return num;
}
