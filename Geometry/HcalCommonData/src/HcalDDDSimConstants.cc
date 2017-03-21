#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

//#define EDM_ML_DEBUG

HcalDDDSimConstants::HcalDDDSimConstants(const HcalParameters* hp) : hpar(hp) {

#ifdef EDM_ML_DEBUG
  edm::LogInfo("HCalGeom") << "HcalDDDSimConstants::HcalDDDSimConstants (const HcalParameter* hp) constructor\n";
#endif

  initialize();
#ifdef EDM_ML_DEBUG
  std::vector<HcalCellType> cellTypes = HcalCellTypes();
  std::cout << "HcalDDDSimConstants: " << cellTypes.size()
	    << " cells of type HCal (All)\n";
#endif
}

HcalDDDSimConstants::~HcalDDDSimConstants() { 
#ifdef EDM_ML_DEBUG
  edm::LogInfo ("HCalGeom") << "HcalDDDSimConstants::destructed!!!\n";
#endif
}

HcalCellType::HcalCell HcalDDDSimConstants::cell(const int idet, const int zside, 
						 const int depth, const int etaR,
						 const int iphi) const {

  double etaMn = hpar->etaMin[0];
  double etaMx = hpar->etaMax[0];
  if (idet==static_cast<int>(HcalEndcap)) {
    etaMn = hpar->etaMin[1]; etaMx = hpar->etaMax[1];
  } else if (idet==static_cast<int>(HcalForward)) {
    etaMn = hpar->etaMin[2]; etaMx = hpar->etaMax[2];
  }
  double eta = 0, deta = 0, phi = 0, dphi = 0, rz = 0, drz = 0;
  bool   ok = false, flagrz = true;
  if ((idet==static_cast<int>(HcalBarrel)||idet==static_cast<int>(HcalEndcap)||
       idet==static_cast<int>(HcalOuter)||idet==static_cast<int>(HcalForward))
      && etaR >=etaMn && etaR <= etaMx && depth > 0)    ok = true;
  if (idet == static_cast<int>(HcalEndcap) && depth>(int)(hpar->zHE.size()))ok=false;
  else if (idet == static_cast<int>(HcalBarrel) && depth > maxLayerHB_+1)   ok=false;
  else if (idet == static_cast<int>(HcalOuter) && depth != 4)               ok=false;
  else if (idet == static_cast<int>(HcalForward) && depth > maxDepth[2])    ok=false;
  if (ok) {
    eta  = getEta(idet, etaR, zside, depth);
    deta = deltaEta(idet, etaR, depth);
    double fibin, fioff;
    if      (idet == static_cast<int>(HcalBarrel)||
	     idet == static_cast<int>(HcalOuter)) {
      fioff = hpar->phioff[0];
      fibin = hpar->phibin[etaR-1];
    } else if (idet == static_cast<int>(HcalEndcap)) {
      fioff = hpar->phioff[1];
      fibin = hpar->phibin[etaR-1];
    } else {
      fioff = hpar->phioff[2];
      fibin = hpar->phitable[etaR-hpar->etaMin[2]];
      if (unitPhi(fibin) > 2) fioff = hpar->phioff[4];
    }
    phi  =-fioff + (iphi - 0.5)*fibin;
    dphi = 0.5*fibin;
    if (idet == static_cast<int>(HcalForward)) {
      int ir = nR + hpar->etaMin[2] - etaR - 1;
      if (ir > 0 && ir < nR) {
	rz     = 0.5*(hpar->rTable[ir]+hpar->rTable[ir-1]);
	drz    = 0.5*(hpar->rTable[ir]-hpar->rTable[ir-1]);
      } else {
	ok     = false;
#ifdef EDM_ML_DEBUG
	edm::LogInfo("HCalGeom") << "HcalDDDSimConstants: wrong eta " << etaR 
				 << " ("  << ir << "/" << nR << ") Detector "
				 << idet << std::endl;
#endif
      }
    } else if (etaR <= nEta) {
      int laymin(depth), laymax(depth);
      if (idet == static_cast<int>(HcalOuter)) {
	laymin = (etaR > hpar->noff[2]) ? ((int)(hpar->zHE.size())) : ((int)(hpar->zHE.size()))-1;
	laymax = ((int)(hpar->zHE.size()));
      }
      double d1=0, d2=0;
      if (idet == static_cast<int>(HcalEndcap)) {
	flagrz = false;
	d1     = hpar->zHE[laymin-1] - hpar->dzHE[laymin-1];
	d2     = hpar->zHE[laymax-1] + hpar->dzHE[laymax-1];
      } else {
	d1     = hpar->rHB[laymin-1] - hpar->drHB[laymin-1];
	d2     = hpar->rHB[laymax-1] + hpar->drHB[laymax-1];
      }
      rz     = 0.5*(d2+d1);
      drz    = 0.5*(d2-d1);
    } else {
      ok = false;
      edm::LogWarning("HCalGeom") << "HcalDDDSimConstants: wrong depth " 
				  << depth << " or etaR " << etaR 
				  << " for detector " << idet;
    }
  } else {
    ok = false;
    edm::LogWarning("HCalGeom") << "HcalDDDSimConstants: wrong depth " << depth
				<< " det " << idet;
  }
  HcalCellType::HcalCell tmp(ok,eta,deta,phi,dphi,rz,drz,flagrz);

#ifdef EDM_ML_DEBUG
  std::cout << "HcalDDDSimConstants: det/side/depth/etaR/phi "
	    << idet  << "/" << zside << "/" << depth << "/" 
	    << etaR << "/" << iphi << " Cell Flag " << tmp.ok 
	    << " "  << tmp.eta << " " << tmp.deta << " phi " 
	    << tmp.phi << " " << tmp.dphi << " r(z) " << tmp.rz
	    << " "  << tmp.drz << " " << tmp.flagrz << std::endl;
#endif
  return tmp;
}

int HcalDDDSimConstants::findDepth(const int det, const int eta, 
				   const int phi, const int zside,
				   const int lay) const {

  int depth = (ldmap_.isValid(det,phi,zside)) ? ldmap_.getDepth(det,eta,phi,zside,lay) : -1;
  return depth;
}

unsigned int HcalDDDSimConstants::findLayer(const int layer, const std::vector<HcalParameters::LayerItem>& layerGroup) const {
  
  unsigned int id = layerGroup.size();
  for (unsigned int i = 0; i < layerGroup.size(); i++) {
    if (layer == (int)(layerGroup[i].layer)) {
      id = i;
      break;
    }
  }
  return id;
}

std::vector<std::pair<double,double> > HcalDDDSimConstants::getConstHBHE(const int type) const {

  std::vector<std::pair<double,double> > gcons;
  if (type == 0) {
    for (unsigned int i=0; i<hpar->rHB.size(); ++i) {
      gcons.push_back(std::pair<double,double>(hpar->rHB[i],hpar->drHB[i]));
    }
  } else {
    for (unsigned int i=0; i<hpar->zHE.size(); ++i) {
      gcons.push_back(std::pair<double,double>(hpar->zHE[i],hpar->dzHE[i]));
    }
  }
  return gcons;
}

int HcalDDDSimConstants::getDepthEta16(const int det, const int phi, 
				       const int zside) const {

  int depth = ldmap_.getDepth16(det,phi,zside);
  if (depth < 0) depth = (det == 2) ? depthEta16[1] : depthEta16[0];
#ifdef EDM_ML_DEBUG
  std::cout << "getDepthEta16: " << det << ":" << depth << std::endl;
#endif
  return depth;
}

int HcalDDDSimConstants::getDepthEta16M(const int det) const {
  int depth = (det == 2) ? depthEta16[1] : depthEta16[0];
  std::vector<int> phis;
  int detsp = ldmap_.validDet(phis);
  if (detsp == det) {
    int zside = (phis[0] > 0) ? 1 : -1;
    int iphi  = (phis[0] > 0) ? phis[0] : -phis[0];
    int depthsp = ldmap_.getDepth16(det,iphi,zside);
    if (det == 1 && depthsp > depth) depth = depthsp;
    if (det == 2 && depthsp < depth) depth = depthsp;
  }
  return depth;
}

int HcalDDDSimConstants::getDepthEta29(const int phi, const int zside, 
				       const int i) const {

  int depth = (i == 0) ? ldmap_.getMaxDepthLastHE(2,phi,zside) : -1;
  if (depth < 0) depth = (i == 1) ? depthEta29[1] : depthEta29[0];
  return depth;
}

int HcalDDDSimConstants::getDepthEta29M(const int i, 
					const bool planOne) const {
  int depth = (i == 1) ? depthEta29[1] : depthEta29[0];
  if (i == 0 && planOne) {
    std::vector<int> phis;
    int detsp = ldmap_.validDet(phis);
    if (detsp == 2) {
      int zside = (phis[0] > 0) ? 1 : -1;
      int iphi  = (phis[0] > 0) ? phis[0] : -phis[0];
      int depthsp = ldmap_.getMaxDepthLastHE(2,iphi,zside);
      if (depthsp > depth) depth = depthsp;
    }
  }
  return depth;
}

std::pair<int,double> HcalDDDSimConstants::getDetEta(const double eta, 
						     const int depth) {

  int    hsubdet(0), ieta(0);
  double etaR(0);
  double heta = fabs(eta);
  for (int i = 0; i < nEta; i++)
    if (heta > hpar->etaTable[i]) ieta = i + 1;
  if (heta <= hpar->etaRange[1]) {
    if ((ieta == hpar->etaMin[1] && depth==depthEta16[1]) ||
	(ieta > hpar->etaMax[0])) {
      hsubdet = static_cast<int>(HcalEndcap);
    } else {
      hsubdet = static_cast<int>(HcalBarrel);
    }
    etaR    = eta;
  } else {
    hsubdet = static_cast<int>(HcalForward);
    double theta = 2.*atan(exp(-heta));
    double hR    = zVcal*tan(theta);
    etaR    = (eta >= 0. ? hR : -hR);
  }
  return std::pair<int,double>(hsubdet,etaR);
}

int HcalDDDSimConstants::getEta(const int det, const int lay, 
				const double hetaR) {

  int    ieta(0);
  if (det == static_cast<int>(HcalForward)) { // Forward HCal
    ieta    = hpar->etaMax[2];
    for (int i = nR-1; i > 0; i--)
      if (hetaR < hpar->rTable[i]) ieta = hpar->etaMin[2] + nR - i - 1;
  } else { // Barrel or Endcap
    ieta  = 1;
    for (int i = 0; i < nEta-1; i++)
      if (hetaR > hpar->etaTable[i]) ieta = i + 1;
    if (det == static_cast<int>(HcalBarrel)) {
      if (ieta > hpar->etaMax[0])  ieta = hpar->etaMax[0];
      if (lay == maxLayer_) {
	if (hetaR > etaHO[1] && ieta == hpar->noff[2]) ieta++;
      }
    } else if (det == static_cast<int>(HcalEndcap)) {
      if (ieta <= hpar->etaMin[1]) ieta = hpar->etaMin[1];
    }
  }
  return ieta;
}

std::pair<int,int> HcalDDDSimConstants::getEtaDepth(const int det, int etaR, 
						    int phi, int zside, 
						    int depth, int lay) {

  //Modify the depth index
  if (det == static_cast<int>(HcalForward)) { // Forward HCal
  } else if (det == static_cast<int>(HcalOuter)) {
    depth = 4;
  } else {
    if (lay >= 0) {
      depth= layerGroup(det, etaR, phi, zside, lay-1);
      if (etaR == hpar->noff[0] && lay > 1) {
	int   kphi   = phi + int((hpar->phioff[3]+0.1)/hpar->phibin[etaR-1]);
	kphi         = (kphi-1)%4 + 1;
	if (kphi == 2 || kphi == 3)
	  depth = layerGroup(det, etaR, phi, zside, lay-2);
      }
    } else if (det == static_cast<int>(HcalBarrel)) {
      if (depth>getMaxDepth(det,etaR,phi,zside,false)) 
	depth = getMaxDepth(det,etaR,phi,zside,false);
    }
    if (etaR >= hpar->noff[1] && depth > getDepthEta29(phi,zside,0)) {
      etaR -= getDepthEta29(phi,zside,1);
    } else if (etaR == hpar->etaMin[1]) {
      if (det == static_cast<int>(HcalBarrel)) {
	if (depth > getDepthEta16(det, phi, zside)) 
	  depth = getDepthEta16(det, phi, zside);
      } else {
	if (depth < getDepthEta16(det, phi, zside)) 
	  depth = getDepthEta16(det, phi, zside);
      }
    }
  }
  if ((det == static_cast<int>(HcalEndcap)) && (etaR == 17) && (lay == 1))
    etaR = 18;
  return std::pair<int,int>(etaR,depth);
}

double HcalDDDSimConstants::getEtaHO(double& etaR, double& x, double& y, 
				     double& z) const {

  if (hpar->zHO.size() > 4) {
    double eta  = fabs(etaR);
    double r    = std::sqrt(x*x+y*y);
    if (r > rminHO) {
      double zz = fabs(z);
      if (zz > hpar->zHO[3]) {
	if (eta <= hpar->etaTable[10]) eta = hpar->etaTable[10]+0.001;
      } else if (zz > hpar->zHO[1]) {
	if (eta <= hpar->etaTable[4])  eta = hpar->etaTable[4]+0.001;
      }
    }
    eta = (z >= 0. ? eta : -eta);
#ifdef EDM_ML_DEBUG
    std::cout << "R " << r << " Z " << z << " eta " << etaR 
	      << ":" << eta << std::endl;
    if (eta != etaR) edm::LogInfo ("HCalGeom") << "**** Check *****";
#endif
    return eta;
  } else {
    return etaR;
  }
}

double HcalDDDSimConstants::getLayer0Wt(const int det, const int phi, 
					const int zside) const {

  double wt = ldmap_.getLayer0Wt(det, phi, zside);
  if (wt < 0) wt = (det == 2) ? hpar->Layer0Wt[1] :  hpar->Layer0Wt[0];
  return wt;
}

int HcalDDDSimConstants::getLayerFront(const int det, const int eta, 
				       const int phi, const int zside,
				       const int depth) const {

  int layer = ldmap_.getLayerFront(det, eta, phi, zside, depth);
  if (layer < 0) {
    if (det == 1 || det == 2) {
      layer = 1;
      for (int l=0; l<getLayerMax(eta,depth); ++l) {
	if ((int)(layerGroup(eta-1,l)) == depth+1) {
	  layer = l+1; break;
	}
      }
    } else {
      layer = (eta > hpar->noff[2]) ? maxLayerHB_+1 : maxLayer_;
    }
  }
  return layer;
}

int HcalDDDSimConstants::getLayerBack(const int det, const int eta, 
				      const int phi, const int zside,
				      const int depth) const {

  int layer = ldmap_.getLayerBack(det, eta, phi, zside, depth);
  if (layer < 0) {
    if (det == 1 || det ==2) {
      layer = depths[depth-1][eta-1];
    } else {
      layer = maxLayer_;
    }
  }
  return layer;
}

int HcalDDDSimConstants::getLayerMax(const int eta, const int depth) const {

  int layermx =  ((eta < hpar->etaMin[1]) && depth-1 < maxDepth[0]) ? maxLayerHB_+1 : (int)layerGroupSize(eta-1);
  return layermx;
}

int HcalDDDSimConstants::getMaxDepth(const int det, const int eta, 
				     const int phi, const int zside,
				     const bool partialDetOnly) const {

  int dmax(-1);
  if (partialDetOnly) {
    if (ldmap_.isValid(det,phi,zside)) {
      dmax = ldmap_.getDepths(eta).second;
    }
  } else if (det == 1 || det == 2) {
    if (ldmap_.isValid(det,phi,zside))
      dmax = ldmap_.getDepths(eta).second;
    else if (det == 2)               
      dmax = layerGroup(eta-1,maxLayer_);
    else if (eta == hpar->etaMax[0]) 
      dmax = getDepthEta16(det,phi,zside);
    else                             
      dmax = layerGroup(eta-1,maxLayerHB_);
  } else if (det == 3) { // HF
    dmax = maxHFDepth(zside*eta,phi);
  } else if (det == 4) { // HO
    dmax = maxDepth[3];
  } else {
    dmax = -1;
  }
  return dmax;
}

int HcalDDDSimConstants::getMinDepth(const int det, const int eta, 
				     const int phi, const int zside,
				     const bool partialDetOnly) const {

  int lmin(-1);
  if (partialDetOnly) {
    if (ldmap_.isValid(det,phi,zside)) {
      lmin = ldmap_.getDepths(eta).first;
    }
  } else if (det == 3) { // HF
    lmin = 1;
  } else if (det == 4) { // HO
    lmin = maxDepth[3];
  } else {
    if (ldmap_.isValid(det,phi,zside)) {
      lmin = ldmap_.getDepths(eta).first;
    } else if (layerGroupSize(eta-1) > 0) {
      lmin = (int)(layerGroup(eta-1, 0));
      unsigned int type  = (det == 1) ? 0 : 1;
      if (type == 1 && eta == hpar->etaMin[1]) 
	lmin = getDepthEta16(det, phi, zside);
    } else {
      lmin = 1;
    }
  }
  return lmin;
}

std::pair<int,int> HcalDDDSimConstants::getModHalfHBHE(const int type) const {

  if (type == 0) {
    return std::pair<int,int>(nmodHB,nzHB);
  } else {
    return std::pair<int,int>(nmodHE,nzHE);
  }
}

std::pair<double,double> HcalDDDSimConstants::getPhiCons(const int det, 
							 const int ieta) const {
  
  double fioff(0), fibin(0);
  if (det == static_cast<int>(HcalForward)) { // Forward HCal
    fioff   = hpar->phioff[2];
    fibin   = hpar->phitable[ieta-hpar->etaMin[2]];
    if  (unitPhi(fibin) > 2) {   // HF double-phi  
      fioff = hpar->phioff[4];
    }
  } else { // Barrel or Endcap
    if (det == static_cast<int>(HcalEndcap)) {
      fioff   = hpar->phioff[1];
    } else {
      fioff   = hpar->phioff[0];
    }
    fibin = hpar->phibin[ieta-1];
  }
  return std::pair<double,double>(fioff,fibin);
}

std::vector<std::pair<int,double> >
HcalDDDSimConstants::getPhis(const int subdet, const int ieta) const {

  std::vector<std::pair<int,double> > phis;
  int ietaAbs = (ieta > 0) ? ieta : -ieta;
  std::pair<double,double> ficons = getPhiCons(subdet, ietaAbs);
  int nphi    = int((CLHEP::twopi+0.1*ficons.second)/ficons.second);
  int units   = unitPhi(subdet, ietaAbs);
  for (int ifi = 0; ifi < nphi; ++ifi) {
    double phi =-ficons.first + (ifi+0.5)*ficons.second;
    int iphi   = phiNumber(ifi+1,units);
    phis.push_back(std::pair<int,double>(iphi,phi));
  }
#ifdef EDM_ML_DEBUG
  std::cout << "getPhis: subdet|ieta|iphi " << subdet << "|" << ieta 
	    << " with " << phis.size() << " phi bins" << std::endl;
  for (unsigned int k=0; k<phis.size(); ++k)
    std::cout << "[" << k << "] iphi " << phis[k].first << " phi "
	      << phis[k].second/CLHEP::deg << std::endl;
#endif
  return phis;
}

std::vector<HcalCellType> HcalDDDSimConstants::HcalCellTypes() const{

  std::vector<HcalCellType> cellTypes = HcalCellTypes(HcalBarrel);
#ifdef EDM_ML_DEBUG
  edm::LogInfo ("HCalGeom") << "HcalDDDSimConstants: " << cellTypes.size()
			    << " cells of type HCal Barrel\n";
  for (unsigned int i=0; i<cellTypes.size(); i++)
    edm::LogInfo ("HCalGeom") << "Cell " << i << " " << cellTypes[i] << "\n";
#endif

  std::vector<HcalCellType> hoCells   = HcalCellTypes(HcalOuter);
#ifdef EDM_ML_DEBUG
  edm::LogInfo ("HCalGeom") << "HcalDDDSimConstants: " << hoCells.size()
			    << " cells of type HCal Outer\n";
  for (unsigned int i=0; i<hoCells.size(); i++)
    edm::LogInfo ("HCalGeom") << "Cell " << i << " " << hoCells[i] << "\n";
#endif
  cellTypes.insert(cellTypes.end(), hoCells.begin(), hoCells.end());

  std::vector<HcalCellType> heCells   = HcalCellTypes(HcalEndcap);
#ifdef EDM_ML_DEBUG
  edm::LogInfo ("HCalGeom") << "HcalDDDSimConstants: " << heCells.size()
			    << " cells of type HCal Endcap\n";
  for (unsigned int i=0; i<heCells.size(); i++)
    edm::LogInfo ("HCalGeom") << "Cell " << i << " " << heCells[i] << "\n";
#endif
  cellTypes.insert(cellTypes.end(), heCells.begin(), heCells.end());

  std::vector<HcalCellType> hfCells   = HcalCellTypes(HcalForward);
#ifdef EDM_ML_DEBUG
  edm::LogInfo ("HCalGeom") << "HcalDDDSimConstants: " << hfCells.size()
			    << " cells of type HCal Forward\n";
  for (unsigned int i=0; i<hfCells.size(); i++)
    edm::LogInfo ("HCalGeom") << "Cell " << i << " " << hfCells[i] << "\n";
#endif
  cellTypes.insert(cellTypes.end(), hfCells.begin(), hfCells.end());

  return cellTypes;
}

std::vector<HcalCellType> HcalDDDSimConstants::HcalCellTypes(const HcalSubdetector subdet, const int ieta, const int depthl) const {

  std::vector<HcalCellType> cellTypes;
  if (subdet == HcalForward) {
    if (dzVcal < 0) return cellTypes;
  }

  int    dmin, dmax, indx, nz;
  double hsize = 0;
  switch(subdet) {
  case HcalEndcap:
    dmin = 1; dmax = maxLayer_+1; indx = 1; nz = nzHE;
    break;
  case HcalForward:
    dmin = 1; dmax = (idHF2QIE.size() > 0) ? 2 : maxDepth[2]; indx = 2; nz = 2;
    break;
  case HcalOuter:
    dmin = 4; dmax = 4; indx = 0; nz = nzHB;
    break;
  default:
    dmin = 1; dmax = maxLayerHB_+1; indx = 0; nz = nzHB;
    break;
  }
  if (depthl > 0) dmin = dmax = depthl;
  int ietamin = (ieta>0) ? ieta : hpar->etaMin[indx];
  int ietamax = (ieta>0) ? ieta : hpar->etaMax[indx];
  int phi     = (indx == 2) ? 3 : 1;

  // Get the Cells 
  int subdet0 = static_cast<int>(subdet);
  for (int depth=dmin; depth<=dmax; depth++) {
    int    shift = getShift(subdet, depth);
    double gain  = getGain (subdet, depth);
    if (subdet == HcalForward) {
      if (depth%2 == 1) hsize = dzVcal;
      else              hsize = dzVcal-0.5*dlShort;
    } 
    for (int eta=ietamin; eta<=ietamax; eta++) {
      for (int iz=0; iz<nz; ++iz) {
	int zside = -2*iz + 1;
	HcalCellType::HcalCell temp1 = cell(subdet0,zside,depth,eta,phi);
	if (temp1.ok) {
	  std::vector<std::pair<int,double> > phis = getPhis(subdet0,eta);
	  HcalCellType temp2(subdet, eta, zside, depth, temp1,
			     shift, gain, hsize);
	  double dphi, fioff;
	  std::vector<int> phiMiss2;
	  if      ((subdet == HcalBarrel) || (subdet == HcalOuter)) {
	    fioff = hpar->phioff[0];
	    dphi  = hpar->phibin[eta-1];
	    if (subdet == HcalOuter) {
	      if (eta == hpar->noff[4]) {
		int kk = (iz == 0) ? 7 : (7+hpar->noff[5]);
		for (int miss=0; miss<hpar->noff[5+iz]; miss++) {
		  phiMiss2.push_back(hpar->noff[kk]);
		  kk++;
		}
	      }
	    }
	  } else if (subdet == HcalEndcap) {
	    fioff = hpar->phioff[1];
	    dphi  = hpar->phibin[eta-1];
	  } else {
	    fioff = hpar->phioff[2];
	    dphi  = hpar->phitable[eta-hpar->etaMin[2]];
	    if (unitPhi(dphi) > 2) fioff = hpar->phioff[4];
	  }
	  int unit  = unitPhi(dphi);
	  temp2.setPhi(phis,phiMiss2,fioff,dphi,unit);
	  cellTypes.push_back(temp2);
	  // For HF look at extra cells
	  if ((subdet == HcalForward) && (idHF2QIE.size() > 0)) {
	    HcalCellType temp3(subdet, eta, zside+2, depth, temp1,
			       shift, gain, hsize);
	    std::vector<int> phiMiss3;
	    for (unsigned int k=0; k<phis.size(); ++k) {
	      bool ok(false);
	      for (unsigned int l=0; l<idHF2QIE.size(); ++l) {
		if ((eta*zside     == idHF2QIE[l].ieta()) && 
		    (phis[k].first == idHF2QIE[l].iphi())) {
		  ok = true;
		  break;
		}
	      }
	      if (!ok) phiMiss3.push_back(phis[k].first);
	    }
	    dphi  = hpar->phitable[eta-hpar->etaMin[2]];
	    unit  = unitPhi(dphi);
	    fioff = (unit > 2) ? hpar->phioff[4] : hpar->phioff[2];
	    temp3.setPhi(phis,phiMiss2,fioff,dphi,unit);
	    cellTypes.push_back(temp3);
	  }
	}
      }
    }
  }
  return cellTypes;
}

int HcalDDDSimConstants::maxHFDepth(const int eta, const int iphi) const {
  int mxdepth = maxDepth[2];
  if (idHF2QIE.size() > 0) {
    bool ok(false);
    for (unsigned int k=0; k<idHF2QIE.size(); ++k) {
      if ((eta == idHF2QIE[k].ieta()) && (iphi == idHF2QIE[k].iphi())) {
	ok = true; break;
      }
    }
    if (!ok) mxdepth = 2;
  }
  return mxdepth;
}

unsigned int HcalDDDSimConstants::numberOfCells(const HcalSubdetector subdet) const{

  unsigned int num = 0;
  std::vector<HcalCellType> cellTypes = HcalCellTypes(subdet);
  for (unsigned int i=0; i<cellTypes.size(); i++) {
    num += (unsigned int)(cellTypes[i].nPhiBins());
  }
#ifdef EDM_ML_DEBUG
  std::cout << "HcalDDDSimConstants:numberOfCells " 
	    << cellTypes.size()  << " " << num 
	    << " for subdetector " << subdet << std::endl;
#endif
  return num;
}

int HcalDDDSimConstants::phiNumber(const int phi, const int units) const {

  int iphi_skip = phi;
  if      (units==2) iphi_skip  = (phi-1)*2+1;
  else if (units==4) iphi_skip  = (phi-1)*4-1;
  if (iphi_skip < 0) iphi_skip += 72;
  return iphi_skip;
}

void HcalDDDSimConstants::printTiles() const {
 
  std::vector<int> phis;
  int detsp = ldmap_.validDet(phis);
  int kphi  = (detsp > 0) ? phis[0] : 1;
  int zside = (kphi > 0) ? 1 : -1;
  int iphi  = (kphi > 0) ? kphi : -kphi;
  std::cout << "Tile Information for HB from " << hpar->etaMin[0] << " to " << hpar->etaMax[0] << "\n\n";
  for (int eta=hpar->etaMin[0]; eta<= hpar->etaMax[0]; eta++) {
    int dmax = getMaxDepth(1,eta,iphi,-zside,false);
    for (int depth=1; depth<=dmax; depth++) 
      printTileHB(eta, iphi, -zside, depth);
    if (detsp == 1) {
      int dmax = getMaxDepth(1,eta,iphi,zside,false);
      for (int depth=1; depth<=dmax; depth++) 
	printTileHB(eta, iphi, zside, depth);
    }
  }

  std::cout << "\nTile Information for HE from " << hpar->etaMin[1] << " to " << hpar->etaMax[1] << "\n\n";
  for (int eta=hpar->etaMin[1]; eta<= hpar->etaMax[1]; eta++) {
    int dmin = (eta == hpar->etaMin[1]) ? getDepthEta16(2,iphi,-zside) : 1;
    int dmax = getMaxDepth(2,eta,iphi,-zside,false);
    for (int depth=dmin; depth<=dmax; depth++)
      printTileHE(eta, iphi, -zside, depth);
    if (detsp == 2) {
      int dmax = getMaxDepth(2,eta,iphi,zside,false);
      for (int depth=1; depth<=dmax; depth++) 
	printTileHE(eta, iphi, zside, depth);
    }
  }
}

int HcalDDDSimConstants::unitPhi(const int det, const int etaR) const {

  double dphi = (det == static_cast<int>(HcalForward)) ? hpar->phitable[etaR-hpar->etaMin[2]] : hpar->phibin[etaR-1];
  return unitPhi(dphi);
}

int HcalDDDSimConstants::unitPhi(const double dphi) const {

  const double fiveDegInRad = 2*M_PI/72;
  int units = int(dphi/fiveDegInRad+0.5);
  if (units < 1) units = 1;
  return units;
}

void HcalDDDSimConstants::initialize( void ) {

  nEta      = hpar->etaTable.size();
  nR        = hpar->rTable.size();
  nPhiF     = nR - 1;
  isBH_     = false;

#ifdef EDM_ML_DEBUG
  for (int i=0; i<nEta-1; ++i) {
    std::cout << "HcalDDDSimConstants:Read LayerGroup" << i << ":";
    for (unsigned int k=0; k<layerGroupSize( i ); k++) 
      std::cout << " [" << k << "] = " << layerGroup( i, k );
    std::cout << std::endl;
  }
#endif

  // Geometry parameters for HF
  dlShort   = hpar->gparHF[0];
  zVcal     = hpar->gparHF[4];
  dzVcal    = hpar->dzVcal;
#ifdef EDM_ML_DEBUG
  std::cout << "HcalDDDSimConstants: dlShort " << dlShort << " zVcal " << zVcal
	    << " and dzVcal " << dzVcal << std::endl;
#endif

  //Transform some of the parameters
  maxDepth = hpar->maxDepth;
  maxDepth[0] = maxDepth[1] = 0;
  for (int i=0; i<nEta-1; ++i) {
    unsigned int imx = layerGroupSize( i );
    int laymax = (imx > 0) ? layerGroup( i, imx-1 ) : 0;
    if (i < hpar->etaMax[0]) {
      int laymax0 = (imx > 16) ? layerGroup(i, 16) : laymax;
      if (i+1 == hpar->etaMax[0] && laymax0 > 2) laymax0 = 2;
      if (maxDepth[0] < laymax0) maxDepth[0] = laymax0;
    }
    if (i >= hpar->etaMin[1]-1 && i < hpar->etaMax[1]) {
      if (maxDepth[1] < laymax) maxDepth[1] = laymax;
    }
  }
#ifdef EDM_ML_DEBUG
  for (int i=0; i<4; ++i)
    std::cout << "Detector Type [" << i << "] iEta " << hpar->etaMin[i] << ":" 
	      << hpar->etaMax[i] << " MaxDepth " << maxDepth[i] << std::endl;
#endif

  int maxdepth = (maxDepth[1]>maxDepth[0]) ? maxDepth[1] : maxDepth[0];
  for (int i=0; i<maxdepth; ++i) {
    for (int k=0; k<nEta-1; ++k) {
      int layermx = getLayerMax(k+1,i+1);
      int ll      = layermx;
      for (int l=layermx-1; l >= 0; --l) {
	if ((int)layerGroup( k, l ) == i+1) {
	  ll = l+1; break;
	}
      }
      depths[i].push_back(ll);
    }

#ifdef EDM_ML_DEBUG
    std::cout << "Depth " << i << " with " << depths[i].size() << " etas:";
    for (int k=0; k<nEta-1; ++k) std::cout << " " << depths[i][k];
    std::cout << std::endl;
#endif
  }

  nzHB   = hpar->modHB[1];
  nmodHB = hpar->modHB[0];
  nzHE   = hpar->modHE[1];
  nmodHE = hpar->modHE[0];
#ifdef EDM_ML_DEBUG
  std::cout << "HcalDDDSimConstants:: " << nzHB << ":" << nmodHB
	    << " barrel and " << nzHE << ":" << nmodHE
	    << " endcap half-sectors" << std::endl;
#endif

  if (hpar->rHB.size() > maxLayerHB_+1 && hpar->zHO.size() > 4) {
    rminHO   = hpar->rHO[0];
    for (int k=0; k<4; ++k) etaHO[k] = hpar->rHO[k+1];
  } else {
    rminHO   =-1.0;
    etaHO[0] = hpar->etaTable[4];
    etaHO[1] = hpar->etaTable[4];
    etaHO[2] = hpar->etaTable[10];
    etaHO[3] = hpar->etaTable[10];
  }
#ifdef EDM_ML_DEBUG
  std::cout << "HO Eta boundaries " << etaHO[0] << " " << etaHO[1]
	    << " " << etaHO[2] << " " << etaHO[3] << std::endl;
  std::cout << "HO Parameters " << rminHO << " " << hpar->zHO.size();
  for (int i=0; i<4; ++i) std::cout << " eta[" << i << "] = " << etaHO[i];
  for (unsigned int i=0; i<hpar->zHO.size(); ++i) 
    std::cout << " zHO[" << i << "] = " << hpar->zHO[i];
  std::cout << std::endl;
#endif

  int noffsize = 7 + hpar->noff[5] + hpar->noff[6];
  int noffl(noffsize+5);
  if ((int)(hpar->noff.size()) > (noffsize+3)) {
    depthEta16[0] = hpar->noff[noffsize];
    depthEta16[1] = hpar->noff[noffsize+1];
    depthEta29[0] = hpar->noff[noffsize+2];
    depthEta29[1] = hpar->noff[noffsize+3];
    if ((int)(hpar->noff.size()) > (noffsize+4)) {
      noffl += (2*hpar->noff[noffsize+4]);
      if ((int)(hpar->noff.size()) > noffl) isBH_ = (hpar->noff[noffl] > 0);
    }
  } else {
    depthEta16[0] = 2;
    depthEta16[1] = 3;
    depthEta29[0] = 2;
    depthEta29[1] = 1;
  }
#ifdef EDM_ML_DEBUG
  std::cout << "isBH_ " << hpar->noff.size() << ":" << noffsize << ":" 
	    << noffl << ":" << isBH_ << std::endl;
  std::cout << "Depth index at ieta = 16 for HB (max) " << depthEta16[0] 
	    << " HE (min) " << depthEta16[1] << "; max depth for itea = 29 : ("
	    << depthEta29[0] << ":" << depthEta29[1] << ")" << std::endl;
#endif
  
  if ((int)(hpar->noff.size()) > (noffsize+4)) {
    int npair = hpar->noff[noffsize+4];
    int kk    = noffsize+4;
    for (int k=0; k<npair; ++k) {
      idHF2QIE.push_back(HcalDetId(HcalForward,hpar->noff[kk+1],hpar->noff[kk+2],1));
      kk += 2;
    }
  }
#ifdef EDM_ML_DEBUG
  std::cout << idHF2QIE.size() << " detector channels having 2 QIE cards:" 
	    << std::endl;
  for (unsigned int k=0; k<idHF2QIE.size(); ++k) 
    std::cout << " [" << k << "] " << idHF2QIE[k] << std::endl;
#endif

  depthMaxSp_ = std::pair<int,int>(0,0);
  int noffk(noffsize+5);
  if ((int)(hpar->noff.size()) > (noffsize+5)) {
    noffk += (2*hpar->noff[noffsize+4]);
    if ((int)(hpar->noff.size()) > noffk+5) {
      int dtype = hpar->noff[noffk+1];
      int nphi  = hpar->noff[noffk+2];
      int ndeps = hpar->noff[noffk+3];
      int ndp16 = hpar->noff[noffk+4];
      int ndp29 = hpar->noff[noffk+5];
      double wt = 0.1*(hpar->noff[noffk+6]);
      if (dtype == 1 || dtype == 2) {
	if ((int)(hpar->noff.size()) >= (noffk+7+nphi+3*ndeps)) {
	  std::vector<int> ifi, iet, ily, idp;
	  for (int i=0; i<nphi; ++i) ifi.push_back(hpar->noff[noffk+7+i]);
	  for (int i=0; i<ndeps;++i) {
	    iet.push_back(hpar->noff[noffk+7+nphi+3*i]);
	    ily.push_back(hpar->noff[noffk+7+nphi+3*i+1]);
	    idp.push_back(hpar->noff[noffk+7+nphi+3*i+2]);
	  }
#ifdef EDM_ML_DEBUG
	  std::cout << "Initialize HcalLayerDepthMap for Detector " << dtype
		    << " etaMax " << hpar->etaMax[dtype] << " with " << nphi 
		    << " sectors";
	  for (int i=0; i<nphi; ++i) std::cout << ":" << ifi[i];
	  std::cout << " and " << ndeps << " depth sections" << std::endl;
	  for (int i=0; i<ndeps;++i)
	    std::cout << " [" << i << "] " << iet[i] << "  " << ily[i] << "  "
		      << idp[i] << std::endl;
	  std::cout << "Maximum depth for last HE Eta tower " << depthEta29[0]
		    << ":" << ndp16 << ":" << ndp29 << " L0 Wt " 
		    << hpar->Layer0Wt[dtype-1] << ":" << wt << std::endl;
#endif
	  ldmap_.initialize(dtype,hpar->etaMax[dtype-1],ndp16,ndp29,wt,ifi,iet,
			    ily,idp);
	  int zside   = (ifi[0]>0) ? 1 : -1;
	  int iphi    = (ifi[0]>0) ? ifi[0] : -ifi[0];
	  depthMaxSp_ = std::pair<int,int>(dtype,ldmap_.getDepthMax(dtype,iphi,zside));
	}
      }
    }
  }
  if (depthMaxSp_.first == 0) {
    depthMaxSp_ = depthMaxDf_ = std::pair<int,int>(2,maxDepth[1]);
  } else if (depthMaxSp_.first == 1) {
    depthMaxDf_ = std::pair<int,int>(1,maxDepth[0]);
    if (depthMaxSp_.second > maxDepth[0]) maxDepth[0] = depthMaxSp_.second;
  } else {
    depthMaxDf_ = std::pair<int,int>(2,maxDepth[1]);
    if (depthMaxSp_.second > maxDepth[1]) maxDepth[1] = depthMaxSp_.second;
  }
#ifdef EDM_ML_DEBUG
  std::cout << "Detector type and maximum depth for all RBX " 
	    << depthMaxDf_.first << ":" << depthMaxDf_.second
	    << " and for special RBX " << depthMaxSp_.first << ":" 
	    << depthMaxSp_.second << std::endl;
#endif
}

double HcalDDDSimConstants::deltaEta(const int det, const int etaR,
				     const int depth) const {

  double tmp = 0;
  if (det == static_cast<int>(HcalForward)) {
    int ir = nR + hpar->etaMin[2] - etaR - 1;
    if (ir > 0 && ir < nR) {
      double z = zVcal;
      if (depth%2 != 1) z += dlShort;
      tmp = 0.5*(getEta(hpar->rTable[ir-1],z)-getEta(hpar->rTable[ir],z));
    }
  } else {
    if (etaR > 0 && etaR < nEta) {
      if (etaR == hpar->noff[1]-1 && depth > depthEta29[0]) {
	tmp = 0.5*(hpar->etaTable[etaR+1]-hpar->etaTable[etaR-1]);
      } else if (det == static_cast<int>(HcalOuter)) {
	if (etaR == hpar->noff[2]) {
	  tmp = 0.5*(etaHO[0]-hpar->etaTable[etaR-1]);
	} else if (etaR == hpar->noff[2]+1) {
	  tmp = 0.5*(hpar->etaTable[etaR]-etaHO[1]);
	} else if (etaR == hpar->noff[3]) {
	  tmp = 0.5*(etaHO[2]-hpar->etaTable[etaR-1]);
	} else if (etaR == hpar->noff[3]+1) {
	  tmp = 0.5*(hpar->etaTable[etaR]-etaHO[3]);
	} else {
	  tmp = 0.5*(hpar->etaTable[etaR]-hpar->etaTable[etaR-1]);
	}
      } else {
	tmp = 0.5*(hpar->etaTable[etaR]-hpar->etaTable[etaR-1]);
      }
    } 
  }
#ifdef EDM_ML_DEBUG
  std::cout << "HcalDDDSimConstants::deltaEta " << etaR << " " 
	    << depth << " ==> " << tmp << std::endl;
#endif
  return tmp;
}

double HcalDDDSimConstants::getEta(const int det, const int etaR,
				   const int zside, const int depth) const {

  double tmp = 0;
  if (det == static_cast<int>(HcalForward)) {
    int ir = nR + hpar->etaMin[2] - etaR - 1;
    if (ir > 0 && ir < nR) {
      double z = zVcal;
      if (depth%2 != 1) z += dlShort;
      tmp = 0.5*(getEta(hpar->rTable[ir-1],z)+getEta(hpar->rTable[ir],z));
    }
  } else {
    if (etaR > 0 && etaR < nEta) {
      if (etaR == hpar->noff[1]-1 && depth > depthEta29[0]) {
	tmp = 0.5*(hpar->etaTable[etaR+1]+hpar->etaTable[etaR-1]);
      } else if (det == static_cast<int>(HcalOuter)) {
	if (etaR == hpar->noff[2]) {
	  tmp = 0.5*(etaHO[0]+hpar->etaTable[etaR-1]);
	} else if (etaR == hpar->noff[2]+1) {
	  tmp = 0.5*(hpar->etaTable[etaR]+etaHO[1]);
	} else if (etaR == hpar->noff[3]) {
	  tmp = 0.5*(etaHO[2]+hpar->etaTable[etaR-1]);
	} else if (etaR == hpar->noff[3]+1) {
	  tmp = 0.5*(hpar->etaTable[etaR]+etaHO[3]);
	} else {
	  tmp = 0.5*(hpar->etaTable[etaR]+hpar->etaTable[etaR-1]);
	}
      } else {
	tmp = 0.5*(hpar->etaTable[etaR]+hpar->etaTable[etaR-1]);
      }
    }
  } 
  if (zside == 0) tmp = -tmp;
#ifdef EDM_ML_DEBUG
  std::cout << "HcalDDDSimConstants::getEta " << etaR << " " 
	    << zside << " " << depth << " ==> " << tmp << "\n";
#endif
  return tmp;
}
 
double HcalDDDSimConstants::getEta(const double r, const double z) const {

  double tmp = 0;
  if (z != 0) tmp = -log(tan(0.5*atan(r/z)));
#ifdef EDM_ML_DEBUG
  std::cout << "HcalDDDSimConstants::getEta " << r << " " << z 
	    << " ==> " << tmp << std::endl;
#endif
  return tmp;
}

int HcalDDDSimConstants::getShift(const HcalSubdetector subdet,
				  const int depth) const {

  int shift;
  switch(subdet) {
  case HcalEndcap:
    shift = hpar->HEShift[0];
    break;
  case HcalForward:
    shift = hpar->HFShift[(depth-1)%2];
    break;
  case HcalOuter:
    shift = hpar->HBShift[3];
    break;
  default:
    shift = hpar->HBShift[0];
    break;
  }
  return shift;
}

double HcalDDDSimConstants::getGain(const HcalSubdetector subdet, 
				    const int depth) const {

  double gain;
  switch(subdet) {
  case HcalEndcap:
    gain = hpar->HEGains[0];
    break;
  case HcalForward:
    gain = hpar->HFGains[(depth-1)%2];
    break;
  case HcalOuter:
    gain = hpar->HBGains[3];
    break;
  default:
    gain = hpar->HBGains[0];
    break;
  }
  return gain;
}

void HcalDDDSimConstants::printTileHB(const int eta, const int phi,
				      const int zside, const int depth) const {
  std::cout << "HcalDDDSimConstants::printTileHB for eta " << eta << " and depth " << depth << "\n";
  
  double etaL   = hpar->etaTable.at(eta-1);
  double thetaL = 2.*atan(exp(-etaL));
  double etaH   = hpar->etaTable.at(eta);
  double thetaH = 2.*atan(exp(-etaH));
  int    layL   = getLayerFront(1,eta,phi,zside,depth);
  int    layH   = getLayerBack(1,eta,phi,zside,depth);
  std::cout << "\ntileHB:: eta|depth " << zside*eta << "|" << depth << " theta " << thetaH/CLHEP::deg << ":" << thetaL/CLHEP::deg << " Layer " << layL << ":" << layH-1 << "\n";
  for (int lay=layL; lay<layH; ++lay) {
    std::vector<double> area(2,0);
    int kk=0;
    for (unsigned int k=0; k<hpar->layHB.size(); ++k) {
      if (lay == hpar->layHB[k]) {
	double zmin = hpar->rhoxHB[k]*std::cos(thetaL)/std::sin(thetaL);
	double zmax = hpar->rhoxHB[k]*std::cos(thetaH)/std::sin(thetaH);
	double dz   = (std::min(zmax,hpar->dxHB[k]) - zmin);
	if (dz > 0) {
	  area[kk] = dz*hpar->dyHB[k];
	  kk++;
	}
      }
    }
    if (area[0] > 0) std::cout << std::setw(2) << lay << " Area " << std::setw(8) << area[0] << " " << std::setw(8) << area[1] << "\n";
  }
}

void HcalDDDSimConstants::printTileHE(const int eta, const int phi,
				      const int zside, const int depth) const {

  double etaL   = hpar->etaTable[eta-1];
  double thetaL = 2.*atan(exp(-etaL));
  double etaH   = hpar->etaTable[eta];
  double thetaH = 2.*atan(exp(-etaH));
  int    layL   = getLayerFront(2,eta,phi,zside,depth);
  int    layH   = getLayerBack(2,eta,phi,zside,depth);
  double phib  = hpar->phibin[eta-1];
  int nphi = 2;
  if (phib > 6*CLHEP::deg) nphi = 1;
  std::cout << "\ntileHE:: Eta/depth " << zside*eta << "|" << depth << " theta " << thetaH/CLHEP::deg << ":" << thetaL/CLHEP::deg << " Layer " << layL << ":" << layH-1 << " phi " << nphi << "\n";
  for (int lay=layL; lay<layH; ++lay) {
    std::vector<double> area(4,0);
    int kk=0;
    for (unsigned int k=0; k<hpar->layHE.size(); ++k) {
      if (lay == hpar->layHE[k]) {
	double rmin = hpar->zxHE[k]*std::tan(thetaH);
	double rmax = hpar->zxHE[k]*std::tan(thetaL);
	if ((lay != 0 || eta == 18) && 
	    (lay != 1 || (eta == 18 && hpar->rhoxHE[k]-hpar->dyHE[k] > 1000) ||
	     (eta != 18 && hpar->rhoxHE[k]-hpar->dyHE[k] < 1000)) &&
	    rmin+30 < hpar->rhoxHE[k]+hpar->dyHE[k] && 
	    rmax > hpar->rhoxHE[k]-hpar->dyHE[k]) {
	  rmin = std::max(rmin,hpar->rhoxHE[k]-hpar->dyHE[k]);
	  rmax = std::min(rmax,hpar->rhoxHE[k]+hpar->dyHE[k]);
	  double dx1 = rmin*std::tan(phib);
	  double dx2 = rmax*std::tan(phib);
	  double ar1=0, ar2=0;
	  if (nphi == 1) {
	    ar1 = 0.5*(rmax-rmin)*(dx1+dx2-4.*hpar->dx1HE[k]);
	  } else {
	    ar1 = 0.5*(rmax-rmin)*(dx1+dx2-2.*hpar->dx1HE[k]);
	    ar2 = 0.5*(rmax-rmin)*((rmax+rmin)*tan(10.*CLHEP::deg)-4*hpar->dx1HE[k])-ar1;
	  }
	  area[kk]   = ar1;
	  area[kk+2] = ar2;
	  kk++;
	}
      }
    }
    if (area[0] > 0 && area[1] > 0) {
      int lay0 = lay-1;
      if (eta == 18) lay0++;
      if (nphi == 1) {
	std::cout << std::setw(2) << lay0 << " Area " << std::setw(8) << area[0] << " " << std::setw(8) << area[1] << "\n";
      } else {
	std::cout << std::setw(2) << lay0 << " Area " << std::setw(8) << area[0] << " " << std::setw(8) << area[1] << ":" << std::setw(8) << area[2] << " " << std::setw(8) << area[3] << "\n";
      }
    }
  }
}

unsigned int HcalDDDSimConstants::layerGroupSize(const int eta) const {
  unsigned int k = 0;
  for (auto const & it : hpar->layerGroupEtaSim) {
    if (it.layer == (unsigned int)(eta + 1)) {
      return it.layerGroup.size();
    }
    if (it.layer > (unsigned int)(eta + 1)) break;
    k = it.layerGroup.size();
  }
  return k;
}

unsigned int HcalDDDSimConstants::layerGroup(const int eta, 
					     const int i) const {
  unsigned int k = 0;
  for (auto const & it :  hpar->layerGroupEtaSim) {
    if (it.layer == (unsigned int)(eta + 1)) {
      return it.layerGroup.at(i);
    }
    if (it.layer > (unsigned int)(eta + 1)) break;
    k = it.layerGroup.at(i);
  }
  return k;
}
 
unsigned int HcalDDDSimConstants::layerGroup(const int det, const int eta,
					     const int phi, const int zside,
					     const int lay) const {

  int depth0 = findDepth(det,eta,phi,zside,lay);
  unsigned int depth = (depth0 > 0) ? (unsigned int)(depth0) : layerGroup(eta-1,lay);
  return depth;
}
