///////////////////////////////////////////////////////////////////////////////
// File: HcalNumberingFromDDD.cc
// Description: Usage of DDD to get to numbering scheme for hadron calorimeter
///////////////////////////////////////////////////////////////////////////////

#include "Geometry/HcalCommonData/interface/HcalNumberingFromDDD.h"

#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDValue.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <iostream>

//#define DebugLog

HcalNumberingFromDDD::HcalNumberingFromDDD(std::string & name,
					   const DDCompactView & cpv) {
  edm::LogInfo("HCalGeom") << "Creating HcalNumberingFromDDD";
  initialize(name, cpv);
}

HcalNumberingFromDDD::~HcalNumberingFromDDD() {
  edm::LogInfo("HCalGeom") << "Deleting HcalNumberingFromDDD";
}

HcalNumberingFromDDD::HcalID HcalNumberingFromDDD::unitID(int det,
							  const CLHEP::Hep3Vector& point,
							  int depth,
							  int lay) const {


  double hx  = point.x();
  double hy  = point.y();
  double hz  = point.z();
  double hR  = sqrt(hx*hx+hy*hy+hz*hz);
  double htheta = (hR == 0. ? 0. : acos(std::max(std::min(hz/hR,1.0),-1.0)));
  double hsintheta = sin(htheta);
  double hphi = (hR*hsintheta == 0. ? 0. :atan2(hy,hx));
  double heta = (fabs(hsintheta) == 1.? 0. : -log(fabs(tan(htheta/2.))) );

  int    hsubdet=0;
  double etaR;

  //First eta index
  if (det == 5) { // Forward HCal
    hsubdet = static_cast<int>(HcalForward);
    hR      = sqrt(hx*hx+hy*hy);
    etaR    = (heta >= 0. ? hR : -hR);
  } else { // Barrel or Endcap
    etaR    = heta;
    if (det == 3) {
      hsubdet = static_cast<int>(HcalBarrel);
      if (zho.size() > 4) etaR = getEtaHO(heta,hx,hy,hz);
    } else {
      hsubdet = static_cast<int>(HcalEndcap);
    }
  }

#ifdef DebugLog
  LogDebug("HCalGeom") << "HcalNumberingFromDDD: point = " << point << " det "
		       << hsubdet << " eta/R " << etaR << " phi " << hphi;
#endif
  HcalNumberingFromDDD::HcalID tmp = unitID(hsubdet,etaR,hphi,depth,lay);
  return tmp;

}

HcalNumberingFromDDD::HcalID HcalNumberingFromDDD::unitID(double eta,double fi,
							  int depth, 
							  int lay) const {

  int    ieta = 0;
  double heta = fabs(eta);
  for (int i = 0; i < nEta; i++)
    if (heta > etaTable[i]) ieta = i + 1;
  int    hsubdet=0;
  double etaR;
  if (ieta <= etaMin[1]) {
    if ((ieta <= etaMin[1] && depth==3) || ieta > etaMax[0]) {
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

  HcalNumberingFromDDD::HcalID tmp = unitID(hsubdet,etaR,fi,depth,lay);
  return tmp;
}


HcalNumberingFromDDD::HcalID HcalNumberingFromDDD::unitID(int det,
							  double etaR,
							  double phi,
							  int depth,
							  int lay) const {

  int ieta=0;
  double fioff, fibin;
  double hetaR = fabs(etaR);

  //First eta index
  if (det == static_cast<int>(HcalForward)) { // Forward HCal
    fioff   = phioff[2];
    ieta    = etaMax[2];
    for (int i = nR-1; i > 0; i--)
      if (hetaR < rTable[i]) ieta = etaMin[2] + nR - i - 1;
    fibin   = phibin[nEta+ieta-etaMin[2]-1];
    if  (ieta > etaMax[2]-2 ) {   // HF double-phi  
      fioff += 0.5*fibin;
    }
  } else { // Barrel or Endcap
    ieta  = 1;
    for (int i = 0; i < nEta-1; i++)
      if (hetaR > etaTable[i]) ieta = i + 1;
    if (det == static_cast<int>(HcalBarrel)) {
      fioff   = phioff[0];
      if (ieta > etaMax[0])  ieta = etaMax[0];
      if (lay == 18 && nOff.size() > 13) {
	if (hetaR > etaHO[1] && ieta == nOff[13]) ieta++;
      }
    } else {
      fioff   = phioff[1];
      if (ieta <= etaMin[1]) ieta = etaMin[1];
    }
    fibin = phibin[ieta-1];
  }

  int    nphi  = int((CLHEP::twopi+0.1*fibin)/fibin);
  int    zside = etaR>0 ? 1: 0;
  double hphi  = phi+fioff;
  if (hphi < 0)    hphi += CLHEP::twopi;
  int    iphi  = int(hphi/fibin) + 1;
  if (iphi > nphi) iphi = 1;

#ifdef DebugLog
  LogDebug("HCalGeom") << "HcalNumberingFromDDD: etaR = " << etaR << " : "
		       << zside << "/" << ieta << " phi " << hphi << " : "
		       << iphi;
#endif
  HcalNumberingFromDDD::HcalID tmp = unitID(det,zside,depth,ieta,iphi,lay);
  return tmp;

}

HcalNumberingFromDDD::HcalID HcalNumberingFromDDD::unitID(int det, int zside,
							  int depth, int etaR,
							  int phi, 
							  int lay) const {

  //Modify the depth index
  if (det == static_cast<int>(HcalForward)) { // Forward HCal
  } else {
    if (lay >= 0) {
      double fibin = phibin[etaR-1];
      int   depth0 = depth1[etaR-1];
      int   kphi   = phi + int((phioff[3]+0.1)/fibin);
      kphi         = (kphi-1)%4 + 1;
      if (etaR == nOff[0] && (kphi == 2 || kphi == 3)) depth0--;
      if (lay <= depth2[etaR-1]) {
	if (lay <= depth0) depth = 1;
	else               depth = 2;
      } else if (lay <= depth3[etaR-1]) {
	depth = 3;
      } else               depth = 4;
    } else if (det == static_cast<int>(HcalBarrel)) {
      if (depth==3) depth = 2;
    }
    if (det != static_cast<int>(HcalBarrel)) {
      if (etaR <= etaMin[1]) depth = 3;
    }
  }
  if (etaR == nOff[1] && depth > 2 && det == static_cast<int>(HcalEndcap))
    etaR = nOff[1]-1;
  if (det == static_cast<int>(HcalBarrel) && depth == 4) {
    det = static_cast<int>(HcalOuter);
  }

  int units     = unitPhi(det, etaR);
  int iphi_skip = phi;
  if      (units==2) iphi_skip  = (phi-1)*2+1;
  else if (units==4) iphi_skip  = (phi-1)*4-1;
  if (iphi_skip < 0) iphi_skip += 72;

#ifdef DebugLog
  LogDebug("HCalGeom") << "HcalNumberingFromDDD: phi units=" <<  units  
                       <<  "  iphi_skip=" << iphi_skip; 
#endif
  HcalNumberingFromDDD::HcalID tmp(det,zside,depth,etaR,phi,iphi_skip,lay);

#ifdef DebugLog
  LogDebug("HCalGeom") << "HcalNumberingFromDDD: det = " << det << " " 
		       << tmp.subdet << " zside = " << tmp.zside << " depth = "
		       << tmp.depth << " eta/R = " << tmp.etaR << " phi = " 
		       << tmp.phi << " layer = " << tmp.lay;
#endif
  return tmp;
}

HcalCellType::HcalCell HcalNumberingFromDDD::cell(int det, int zside, 
						  int depth, int etaR,
						  int iphi, bool corr) const {

  int idet = det;
  double etaMn = etaMin[0];
  double etaMx = etaMax[0];
  if (idet==static_cast<int>(HcalEndcap)) {
    etaMn = etaMin[1]; etaMx = etaMax[1];
  } else if (idet==static_cast<int>(HcalForward)) {
    etaMn = etaMin[2]; etaMx = etaMax[2];
  }
  if (corr) {
    if (etaR >= nOff[2] && depth == 3 && idet == static_cast<int>(HcalBarrel))
      idet = static_cast<int>(HcalEndcap);
  }
  double eta = 0, deta = 0, phi = 0, dphi = 0, rz = 0, drz = 0;
  bool   ok = false, flagrz = true;
  if ((idet==static_cast<int>(HcalBarrel)||idet==static_cast<int>(HcalEndcap)||
       idet==static_cast<int>(HcalOuter)||idet==static_cast<int>(HcalForward))
      && etaR >=etaMn && etaR <= etaMx)
    ok = true;
  if (idet == static_cast<int>(HcalEndcap)) {
    if      (depth < 3 && etaR <= etaMin[1]) ok = false;
    else if (depth > 2 && etaR == nOff[1])   ok = false;
  }
  if (ok) {
    int maxlay = (int)(rHB.size());
    if (idet == static_cast<int>(HcalEndcap)) maxlay = (int)(zHE.size());
    eta  = getEta(idet, etaR, zside, depth);
    deta = deltaEta(idet, etaR, depth);
    double fibin, fioff;
    if      (idet == static_cast<int>(HcalBarrel)||
	     idet == static_cast<int>(HcalOuter)) {
      fioff = phioff[0];
      fibin = phibin[etaR-1];
    } else if (idet == static_cast<int>(HcalEndcap)) {
      fioff = phioff[1];
      fibin = phibin[etaR-1];
    } else {
      fioff = phioff[2];
      fibin = phibin[nEta+etaR-etaMin[2]-1];
      if  (etaR > etaMax[2]-2 ) fioff += 0.5*fibin; 
    }
    phi  = fioff + (iphi - 0.5)*fibin;
    dphi = 0.5*fibin;
    if (idet == static_cast<int>(HcalForward)) {
      int ir = nR + etaMin[2] - etaR - 1;
      if (ir > 0 && ir < nR) {
	rz     = 0.5*(rTable[ir]+rTable[ir-1]);
	drz    = 0.5*(rTable[ir]-rTable[ir-1]);
      } else {
	ok     = false;
#ifdef DebugLog
	LogDebug("HCalGeom") << "HcalNumberingFromDDD: wrong eta " << etaR 
			     << " ("  << ir << "/" << nR << ") Detector "
			     << idet;
#endif
      }
      if (depth != 1 && depth != 2) {
	ok     = false;
#ifdef DebugLog
	LogDebug("HCalGeom") << "HcalNumberingFromDDD: wrong depth " << depth
			     << " in Detector " << idet;
#endif
      }
    } else if (etaR <= nEta) {
      int depth0 = depth1[etaR-1];
      int kphi   = iphi + int((phioff[3]+0.1)/fibin);
      kphi       = (kphi-1)%4 + 1;
      if (etaR == nOff[0] && (kphi == 2 || kphi == 3)) depth0--;
      int laymin, laymax;
      if (depth == 1) {
	laymin = 1;
	if (idet==static_cast<int>(HcalEndcap)) laymin = 2;
	laymax = depth0;
	if (nOff.size() > 12) {
	  if (etaR == nOff[6]) {
	    laymin = nOff[7];
	    laymax = nOff[8];
	  } else if (etaR == nOff[9]) {
	    laymin = nOff[10];
	  }
	}
      } else if (depth == 2) {
	laymin = depth0+1;
        laymax = depth2[etaR-1];
	if (etaR==etaMax[0] && idet==static_cast<int>(HcalBarrel) &&
	    nOff.size()>3) laymax = nOff[3];
	if (nOff.size() > 12) {
	  if (etaR == nOff[9]) laymax = nOff[11];
	  if (etaR == nOff[6]) laymax = nOff[12];
	}
      } else  if (depth == 3) {
	laymin = depth2[etaR-1]+1;
        laymax = depth3[etaR-1];
	if (etaR<=etaMin[1] && idet==static_cast<int>(HcalEndcap)) {
	  if (nOff.size() > 4) laymin = nOff[4];
	  if (nOff.size() > 5) laymax = nOff[5];
	}
      } else {
	laymin = depth3[etaR-1]+1;
	laymax = maxlay;
      }
      if (idet == static_cast<int>(HcalOuter) && nOff.size() > 13) {
	if (etaR > nOff[13] && laymin <= laymax) laymin = laymax;
      }
      double d1=0, d2=0;
      if (laymin <= maxlay && laymax <= maxlay && laymin <= laymax) {
	if (idet == static_cast<int>(HcalEndcap)) {
	  flagrz = false;
	  if (depth == 1 || laymin <= 1) d1 = zHE[laymin-1] - dzHE[laymin-1];
	  else                           d1 = zHE[laymin-2] + dzHE[laymin-2];
	  d2     = zHE[laymax-1] + dzHE[laymax-1];
	} else {
	  if (idet == static_cast<int>(HcalOuter) ||
	      depth == 1 || laymin <=1) d1 = rHB[laymin-1] - drHB[laymin-1];
	  else                          d1 = rHB[laymin-2] + drHB[laymin-1];
	  d2     = rHB[laymax-1] + drHB[laymax-1];
	}
	rz     = 0.5*(d2+d1);
	drz    = 0.5*(d2-d1);
      } else {
	ok = false;
#ifdef DebugLog
	LogDebug("HCalGeom") << "HcalNumberingFromDDD: wrong depth " << depth
			     << " (Layer minimum " << laymin << " maximum " 
			     << laymax << " maxLay " << maxlay << ")";
#endif
      }
    } else {
      ok = false;
#ifdef DebugLog
      LogDebug("HCalGeom") << "HcalNumberingFromDDD: wrong eta " << etaR
			   << "/" << nEta << " Detector " << idet;
#endif
    }
  } else {
    ok = false;
#ifdef DebugLog
    LogDebug("HCalGeom") << "HcalNumberingFromDDD: wrong eta " << etaR 
			 << " det " << idet;
#endif
  }
  HcalCellType::HcalCell tmp(ok,eta,deta,phi,dphi,rz,drz,flagrz);

#ifdef DebugLog
  LogDebug("HCalGeom") << "HcalNumberingFromDDD: det/side/depth/etaR/phi "
		       << det  << "/" << zside << "/" << depth << "/" << etaR
		       << "/" << iphi << " Cell Flag " << tmp.ok << " " 
		       << tmp.eta << " " << tmp.deta << " phi " << tmp.phi 
		       << " " << tmp.dphi << " r(z) " << tmp.rz  << " " 
		       << tmp.drz << " " << tmp.flagrz;
#endif
  return tmp;
}

std::vector<double> HcalNumberingFromDDD::getEtaTable() const {

  std::vector<double> tmp = etaTable;
  return tmp;
}

unsigned int HcalNumberingFromDDD::numberOfCells(HcalSubdetector subdet) const{

  unsigned int num = 0;
  std::vector<HcalCellType> cellTypes = HcalCellTypes(subdet);
  for (unsigned int i=0; i<cellTypes.size(); i++) {
    num += (unsigned int)(cellTypes[i].nPhiBins());
    if (cellTypes[i].nHalves() > 1) 
      num += (unsigned int)(cellTypes[i].nPhiBins());
    num -= (unsigned int)(cellTypes[i].nPhiMissingBins());
  }
#ifdef DebugLog
  LogDebug ("HCalGeom") << "HcalNumberingFromDDD:numberOfCells " 
			<< cellTypes.size()  << " " << num 
			<< " for subdetector " << subdet;
#endif
  return num;
}

std::vector<HcalCellType> HcalNumberingFromDDD::HcalCellTypes() const{

  std::vector<HcalCellType> cellTypes =HcalCellTypes(HcalBarrel);
#ifdef DebugLog
  LogDebug ("HCalGeom") << "HcalNumberingFromDDD: " << cellTypes.size()
			<< " cells of type HCal Barrel";
  for (unsigned int i=0; i<cellTypes.size(); i++)
    LogDebug ("HCalGeom") << "Cell " << i << " " << cellTypes[i];
#endif

  std::vector<HcalCellType> hoCells   =HcalCellTypes(HcalOuter);
#ifdef DebugLog
  LogDebug ("HCalGeom") << "HcalNumberingFromDDD: " << hoCells.size()
			<< " cells of type HCal Outer";
  for (unsigned int i=0; i<hoCells.size(); i++)
    LogDebug ("HCalGeom") << "Cell " << i << " " << hoCells[i];
#endif
  cellTypes.insert(cellTypes.end(), hoCells.begin(), hoCells.end());

  std::vector<HcalCellType> heCells   =HcalCellTypes(HcalEndcap);
#ifdef DebugLog
  LogDebug ("HCalGeom") << "HcalNumberingFromDDD: " << heCells.size()
			<< " cells of type HCal Endcap";
  for (unsigned int i=0; i<heCells.size(); i++)
    LogDebug ("HCalGeom") << "Cell " << i << " " << heCells[i];
#endif
  cellTypes.insert(cellTypes.end(), heCells.begin(), heCells.end());

  std::vector<HcalCellType> hfCells   =HcalCellTypes(HcalForward);
#ifdef DebugLog
  LogDebug ("HCalGeom") << "HcalNumberingFromDDD: " << hfCells.size()
			<< " cells of type HCal Forward";
  for (unsigned int i=0; i<hfCells.size(); i++)
    LogDebug ("HCalGeom") << "Cell " << i << " " << hfCells[i];
#endif
  cellTypes.insert(cellTypes.end(), hfCells.begin(), hfCells.end());

  return cellTypes;
}

std::vector<HcalCellType> HcalNumberingFromDDD::HcalCellTypes(HcalSubdetector subdet) const {

  std::vector<HcalCellType> cellTypes;
  if (subdet == HcalForward) {
    if (dzVcal < 0) return cellTypes;
  }

  int    dmin, dmax, indx, nz, nmod;
  double hsize = 0;
  switch(subdet) {
  case HcalEndcap:
    dmin = 1; dmax = 3; indx = 1; nz = nzHE; nmod = nmodHE;
    break;
  case HcalForward:
    dmin = 1; dmax = 2; indx = 2; nz = 2; nmod = 18; 
    break;
  case HcalOuter:
    dmin = 4; dmax = 4; indx = 0; nz = nzHB; nmod = nmodHB;
    break;
  default:
    dmin = 1; dmax = 3; indx = 0; nz = nzHB; nmod = nmodHB;
    break;
  }

  int phi = 1, zside  = 1;
  bool cor = false;

  // Get the Cells 
  int subdet0 = static_cast<int>(subdet);
  for (int depth=dmin; depth<=dmax; depth++) {
    int    shift = getShift(subdet, depth);
    double gain  = getGain (subdet, depth);
    if (subdet == HcalForward) {
      if (depth == 1) hsize = dzVcal;
      else            hsize = dzVcal-0.5*dlShort;
    }
    for (int eta=etaMin[indx]; eta<= etaMax[indx]; eta++) {
      HcalCellType::HcalCell temp1 = cell(subdet0,zside,depth,eta,phi,cor);
      if (temp1.ok) {
	int units = unitPhi (subdet0, eta);
	HcalCellType temp2(subdet, eta, phi, depth, temp1,
					 shift, gain, nz, nmod, hsize, units);
	if (subdet == HcalOuter && nOff.size() > 17) {
	  if (eta == nOff[15]) {
	    std::vector<int> missPlus, missMinus;
	    int kk = 18;
	    for (int miss=0; miss<nOff[16]; miss++) {
	      missPlus.push_back(nOff[kk]);
	      kk++;
	    }
	    for (int miss=0; miss<nOff[17]; miss++) {
	      missMinus.push_back(nOff[kk]);
	      kk++;
	    }
	    temp2.setMissingPhi(missPlus, missMinus);
	  }
	}
	cellTypes.push_back(temp2);
      }
    }
  }
  return cellTypes;
}

void HcalNumberingFromDDD::printTile() {
 
  std::cout << "Tile Information for HB:\n" << "========================\n\n";
  for (int eta=etaMin[0]; eta<= etaMax[0]; eta++) {
    int dmax = 1;
    if (depth1[eta-1] < 17) dmax = 2;
    for (int depth=1; depth<=dmax; depth++) 
      tileHB(eta, depth);
  }

  std::cout << "\nTile Information for HE:\n" <<"========================\n\n";
  for (int eta=etaMin[1]; eta<= etaMax[1]; eta++) {
    int dmin=1, dmax=3;
    if (eta == etaMin[1]) {
      dmin = 3;
    } else if (depth1[eta-1] > 18) {
      dmax = 1;
    } else if (depth2[eta-1] > 18) {
      dmax = 2;
    }
    for (int depth=dmin; depth<=dmax; depth++)
      tileHE(eta, depth);
  }
}

double HcalNumberingFromDDD::getEta(int det, int etaR, int zside,
				    int depth) const {

  double tmp = 0;
  if (det == static_cast<int>(HcalForward)) {
    int ir = nR + etaMin[2] - etaR - 1;
    if (ir > 0 && ir < nR) {
      double z = zVcal;
      if (depth != 1) z += dlShort;
      tmp = 0.5*(getEta(rTable[ir-1],z)+getEta(rTable[ir],z));
    }
  } else {
    if (etaR > 0 && etaR < nEta) {
      if (etaR == nOff[1]-1 && depth > 2) {
	tmp = 0.5*(etaTable[etaR+1]+etaTable[etaR-1]);
      } else if (det == static_cast<int>(HcalOuter) && nOff.size() > 13) {
	if (etaR == nOff[13]) {
	  tmp = 0.5*(etaHO[0]+etaTable[etaR-1]);
	} else if (etaR == nOff[13]+1) {
	  tmp = 0.5*(etaTable[etaR]+etaHO[1]);
	} else if (etaR == nOff[14]) {
	  tmp = 0.5*(etaHO[2]+etaTable[etaR-1]);
	} else if (etaR == nOff[14]+1) {
	  tmp = 0.5*(etaTable[etaR]+etaHO[3]);
	} else {
	  tmp = 0.5*(etaTable[etaR]+etaTable[etaR-1]);
	}
      } else {
	tmp = 0.5*(etaTable[etaR]+etaTable[etaR-1]);
      }
    }
  } 
  if (zside == 0) tmp = -tmp;
#ifdef DebugLog
  LogDebug("HCalGeom") << "HcalNumberingFromDDD::getEta " << etaR << " " 
		       << zside << " " << depth << " ==> " << tmp;
#endif
  return tmp;
}
 
double HcalNumberingFromDDD::getEta(double r, double z) const {

  double tmp = 0;
  if (z != 0) tmp = -log(tan(0.5*atan(r/z)));
#ifdef DebugLog
  LogDebug("HCalGeom") << "HcalNumberingFromDDD::getEta " << r << " " << z 
		       << " ==> " << tmp;
#endif
  return tmp;
}

double HcalNumberingFromDDD::deltaEta(int det, int etaR, int depth) const {

  double tmp = 0;
  if (det == static_cast<int>(HcalForward)) {
    int ir = nR + etaMin[2] - etaR - 1;
    if (ir > 0 && ir < nR) {
      double z = zVcal;
      if (depth != 1) z += dlShort;
      tmp = 0.5*(getEta(rTable[ir-1],z)-getEta(rTable[ir],z));
    }
  } else {
    if (etaR > 0 && etaR < nEta) {
      if (etaR == nOff[1]-1 && depth > 2) {
	tmp = 0.5*(etaTable[etaR+1]-etaTable[etaR-1]);
      } else if (det == static_cast<int>(HcalOuter) && nOff.size() > 13) {
	if (etaR == nOff[13]) {
	  tmp = 0.5*(etaHO[0]-etaTable[etaR-1]);
	} else if (etaR == nOff[13]+1) {
	  tmp = 0.5*(etaTable[etaR]-etaHO[1]);
	} else if (etaR == nOff[14]) {
	  tmp = 0.5*(etaHO[2]-etaTable[etaR-1]);
	} else if (etaR == nOff[14]+1) {
	  tmp = 0.5*(etaTable[etaR]-etaHO[3]);
	} else {
	  tmp = 0.5*(etaTable[etaR]-etaTable[etaR-1]);
	}
      } else {
	tmp = 0.5*(etaTable[etaR]-etaTable[etaR-1]);
      }
    } 
  }
#ifdef DebugLog
  LogDebug("HCalGeom") << "HcalNumberingFromDDD::deltaEta " << etaR << " " 
		       << depth << " ==> " << tmp;
#endif
  return tmp;
}

void HcalNumberingFromDDD::initialize(std::string & name, 
				      const DDCompactView & cpv) {

  std::string attribute = "ReadOutName";
  edm::LogInfo("HCalGeom") << "HcalNumberingFromDDD: Initailise for " << name 
			   << " as " << attribute;

  DDSpecificsFilter filter;
  DDValue           ddv(attribute,name,0);
  filter.setCriteria(ddv,DDSpecificsFilter::equals);
  DDFilteredView fv(cpv);
  fv.addFilter(filter);
  bool ok = fv.firstChild();

  if (ok) {
    //Load the SpecPars
    loadSpecPars(fv);

    //Load the Geometry parameters
    loadGeometry(fv);
  } else {
    edm::LogError("HCalGeom") << "HcalNumberingFromDDD: cannot get filtered "
			      << " view for " << attribute << " matching "
			      << name;
    throw cms::Exception("DDException") << "HcalNumberingFromDDD: cannot match " << attribute << " to " << name;
  }

#ifdef DebugLog
  std::vector<HcalCellType> cellTypes = HcalCellTypes();
  LogDebug ("HCalGeom") << "HcalNumberingFromDDD: " << cellTypes.size()
			<< " cells of type HCal (All)";
  for (unsigned int i=0; i<cellTypes.size(); i++)
    LogDebug ("HCalGeom") << "Cell " << i << " " << cellTypes[i];
#endif
}

void HcalNumberingFromDDD::loadSpecPars(const DDFilteredView& fv) {

  DDsvalues_type sv(fv.mergedSpecifics());

  // Phi Offset
  int i, nphi=4;
  std::vector<double> tmp1 = getDDDArray("phioff",sv,nphi);
  phioff.resize(tmp1.size());
  for (i=0; i<nphi; i++) {
    phioff[i] = tmp1[i];
#ifdef DebugLog
    LogDebug("HCalGeom") << "HcalNumberingFromDDD: phioff[" << i << "] = "
			 << phioff[i]/CLHEP::deg;
#endif
  }

  //Eta table
  nEta     = -1;
  std::vector<double> tmp2 = getDDDArray("etaTable",sv,nEta);
  etaTable.resize(tmp2.size());
  for (i=0; i<nEta; i++) {
    etaTable[i] = tmp2[i];
#ifdef DebugLog
    LogDebug("HCalGeom") << "HcalNumberingFromDDD: etaTable[" << i << "] = "
			 << etaTable[i];
#endif
  }

  //R table
  nR     = -1;
  std::vector<double> tmp3 = getDDDArray("rTable",sv,nR);
  rTable.resize(tmp3.size());
  for (i=0; i<nR; i++) {
    rTable[i] = tmp3[i];
#ifdef DebugLog
    LogDebug("HCalGeom") << "HcalNumberingFromDDD: rTable[" << i << "] = "
			 << rTable[i]/CLHEP::cm;
#endif
  }

  //Phi bins
  nPhi   = nEta + nR - 2;
  std::vector<double> tmp4 = getDDDArray("phibin",sv,nPhi);
  phibin.resize(tmp4.size());
  for (i=0; i<nPhi; i++) {
    phibin[i] = tmp4[i];
#ifdef DebugLog
    LogDebug("HCalGeom") << "HcalNumberingFromDDD: phibin[" << i << "] = "
			 << phibin[i]/CLHEP::deg;
#endif
  }

  //Layer boundaries for depths 1, 2, 3, 4
  nDepth            = nEta - 1;
  std::vector<double> d1 = getDDDArray("depth1",sv,nDepth);
  nDepth            = nEta - 1;
  std::vector<double> d2 = getDDDArray("depth2",sv,nDepth);
  nDepth            = nEta - 1;
  std::vector<double> d3 = getDDDArray("depth3",sv,nDepth);
#ifdef DebugLog
  LogDebug("HCalGeom") << "HcalNumberingFromDDD: " << nDepth << " Depths";
#endif
  depth1.resize(nDepth);
  depth2.resize(nDepth);
  depth3.resize(nDepth);
  for (i=0; i<nDepth; i++) {
    depth1[i] = static_cast<int>(d1[i]);
    depth2[i] = static_cast<int>(d2[i]);
    depth3[i] = static_cast<int>(d3[i]);
#ifdef DebugLog
    LogDebug("HCalGeom") << "HcalNumberingFromDDD: depth1[" << i << "] = " 
			 << depth1[i] << " depth2[" << i << "]  = "<< depth2[i]
			 << " depth3[" << i << "] = " << depth3[i];
#endif
  }

  // Minimum and maximum eta boundaries
  int ndx  = 3;
  std::vector<double>  tmp5 = getDDDArray("etaMin",sv,ndx);
  std::vector<double>  tmp6 = getDDDArray("etaMax",sv,ndx);
  etaMin.resize(ndx);
  etaMax.resize(ndx);
  for (i=0; i<ndx; i++) {
    etaMin[i] = static_cast<int>(tmp5[i]);
    etaMax[i] = static_cast<int>(tmp6[i]);
  }
  etaMin[0] = 1;
  etaMax[1] = nEta-1;
  etaMax[2] = etaMin[2]+nR-2;
#ifdef DebugLog
  for (i=0; i<ndx; i++) 
    LogDebug("HCalGeom") << "HcalNumberingFromDDD: etaMin[" << i << "] = "
			 << etaMin[i] << " etaMax[" << i << "] = "<< etaMax[i];
#endif

  // Geometry parameters for HF
  int ngpar = 7;
  std::vector<double> gpar = getDDDArray("gparHF",sv,ngpar);
  dlShort = gpar[0];
  zVcal   = gpar[4];
#ifdef DebugLog
  LogDebug("HCalGeom") << "HcalNumberingFromDDD: dlShort " << dlShort
		       << " zVcal " << zVcal;
#endif

  // nOff
  int noff = 3;
  std::vector<double>  nvec = getDDDArray("noff",sv,noff);
  nOff.resize(noff);
  for (i=0; i<noff; i++) {
    nOff[i] = static_cast<int>(nvec[i]);
#ifdef DebugLog
    LogDebug("HCalGeom") << "HcalNumberingFromDDD: nOff[" << i << "] = " 
			 << nOff[i];
#endif
  }

  //Gains and Shifts for HB depths
  ndx                  = 4;
  gainHB               = getDDDArray("HBGains",sv,ndx);
  std::vector<double>  tmp7 = getDDDArray("HBShift",sv,ndx);
  shiftHB.resize(ndx);
#ifdef DebugLog
  LogDebug("HCalGeom") << "HcalNumberingFromDDD:: Gain factor and Shift for "
		       << "HB depth layers:";
#endif
  for (i=0; i<ndx; i++) {
    shiftHB[i] = static_cast<int>(tmp7[i]);
#ifdef DebugLog
    LogDebug("HCalGeom") <<"HcalNumberingFromDDD:: gainHB[" <<  i << "] = " 
			 << gainHB[i] << " shiftHB[" << i << "] = " 
			 << shiftHB[i];
#endif
  }

  //Gains and Shifts for HB depths
  ndx                  = 4;
  gainHE               = getDDDArray("HEGains",sv,ndx);
  std::vector<double>  tmp8 = getDDDArray("HEShift",sv,ndx);
  shiftHE.resize(ndx);
#ifdef DebugLog
  LogDebug("HCalGeom") << "HcalNumberingFromDDD:: Gain factor and Shift for "
		       << "HE depth layers:";
#endif
  for (i=0; i<ndx; i++) {
    shiftHE[i] = static_cast<int>(tmp8[i]);
#ifdef DebugLog
    LogDebug("HCalGeom") <<"HcalNumberingFromDDD:: gainHE[" <<  i << "] = " 
			 << gainHE[i] << " shiftHE[" << i << "] = " 
			 << shiftHE[i];
#endif
  }

  //Gains and Shifts for HF depths
  ndx                  = 4;
  gainHF               = getDDDArray("HFGains",sv,ndx);
  std::vector<double>  tmp9 = getDDDArray("HFShift",sv,ndx);
  shiftHF.resize(ndx);
#ifdef DebugLog
  LogDebug("HCalGeom") << "HcalNumberingFromDDD:: Gain factor and Shift for "
		       << "HF depth layers:";
#endif
  for (i=0; i<ndx; i++) {
    shiftHF[i] = static_cast<int>(tmp9[i]);
#ifdef DebugLog
    LogDebug("HCalGeom") <<"HcalNumberingFromDDD:: gainHF[" <<  i << "] = " 
			 << gainHF[i] << " shiftHF[" << i << "] = " 
			 << shiftHF[i];
#endif
  }
}

void HcalNumberingFromDDD::loadGeometry(const DDFilteredView& _fv) {

  DDFilteredView fv = _fv;
  bool dodet=true, hf=false;
  std::vector<double> rb(20,0.0), ze(20,0.0), thkb(20,-1.0), thke(20,-1.0);
  std::vector<int>    ib(20,0),   ie(20,0);
  std::vector<int>    izb, phib, ize, phie, izf, phif;
  std::vector<double> rxb;
  rhoxb.clear(); zxb.clear(); dyxb.clear(); dzxb.clear();
  layb.clear(); laye.clear();
  zxe.clear(); rhoxe.clear(); dyxe.clear(); dx1e.clear(); dx2e.clear();
  double zf = 0;
  dzVcal = -1.;

  while (dodet) {
    DDTranslation    t    = fv.translation();
    std::vector<int> copy = fv.copyNumbers();
    const DDSolid & sol  = fv.logicalPart().solid();
    int idet = 0, lay = -1;
    int nsiz = (int)(copy.size());
    if (nsiz>0) lay  = copy[nsiz-1]/10;
    if (nsiz>1) idet = copy[nsiz-2]/1000;
    double dx=0, dy=0, dz=0, dx1=0, dx2=0;
    if (sol.shape() == 1) {
      const DDBox & box = static_cast<DDBox>(fv.logicalPart().solid());
      dx = box.halfX();
      dy = box.halfY();
      dz = box.halfZ();
    } else if (sol.shape() == 3) {
      const DDTrap & trp = static_cast<DDTrap>(fv.logicalPart().solid());
      dx1= trp.x1();
      dx2= trp.x2();
      dx = 0.25*(trp.x1()+trp.x2()+trp.x3()+trp.x4());
      dy = 0.5*(trp.y1()+trp.y2());
      dz = trp.halfZ();
    } else if (sol.shape() == 2) {
      const DDTubs & tub = static_cast<DDTubs>(fv.logicalPart().solid());
      dx = tub.rIn();
      dy = tub.rOut();
      dz = tub.zhalf();
    }
    if (idet == 3) {
      // HB
#ifdef DebugLog
      LogDebug("HCalGeom") << "HB " << sol.name() << " Shape " << sol.shape()
			   << " Layer " << lay << " R " << t.Rho();
#endif
      if (lay >=0 && lay < 20) {
	ib[lay]++;
	rb[lay] += t.Rho();
	if (thkb[lay] <= 0) {
	  if (lay < 17) thkb[lay] = dx;
	  else          thkb[lay] = std::min(dx,dy);
	}
	if (lay < 17) {
	  bool found = false;
	  for (unsigned int k=0; k<rxb.size(); k++) {
	    if (std::abs(rxb[k]-t.Rho()) < 0.01) {
	      found = true;
	      break;
	    }
	  }
	  if (!found) {
	    rxb.push_back(t.Rho());
	    rhoxb.push_back(t.Rho()*std::cos(t.phi()));
	    zxb.push_back(std::abs(t.z()));
	    dyxb.push_back(2.*dy);
	    dzxb.push_back(2.*dz);
	    layb.push_back(lay);
	  }
	}
      }
      if (lay == 2) {
	int iz = copy[nsiz-5];
	int fi = copy[nsiz-4];
	unsigned int it1 = find(iz, izb);
	if (it1 == izb.size())  izb.push_back(iz);
	unsigned int it2 = find(fi, phib);
	if (it2 == phib.size()) phib.push_back(fi);
      }
      if (lay == 18) {
	int ifi=-1, ich=-1;
	if (nsiz>2) ifi = copy[nsiz-3];
	if (nsiz>3) ich = copy[nsiz-4];
	double z1 = std::abs((t.z()) + dz);
	double z2 = std::abs((t.z()) - dz);
	if (std::abs(z1-z2) < 0.01) z1 = 0;
        if (ifi == 1 && ich == 4) {
	  if (z1 > z2) {
	    double tmp = z1;
	    z1 = z2;
	    z2 = tmp;
	  }
	  bool sok = true;
	  for (unsigned int kk=0; kk<zho.size(); kk++) {
	    if (std::abs(z2-zho[kk]) < 0.01) {
	      sok = false;
	      break;
	    }	else if (z2 < zho[kk]) {
	      zho.resize(zho.size()+2);
	      for (unsigned int kz=zho.size()-1; kz>kk+1; kz=kz-2) {
		zho[kz]   = zho[kz-2];
		zho[kz-1] = zho[kz-3];
	      }
	      zho[kk+1] = z2;
	      zho[kk]   = z1;
	      sok = false;
	      break;
	    }
	  }
	  if (sok) {
	    zho.push_back(z1);
	    zho.push_back(z2);
	  }
#ifdef DebugLog
	  LogDebug("HCalGeom") << "Detector " << idet << " Lay " << lay << " fi " << ifi << " " << ich << " z " << z1 << " " << z2;
#endif
	}
      }
    } else if (idet == 4) {
      // HE
#ifdef DebugLog
      LogDebug("HCalGeom") << "HE " << sol.name() << " Shape " << sol.shape()
			   << " Layer " << lay << " Z " << t.z();
#endif
      if (lay >=0 && lay < 20) {
	ie[lay]++;
	ze[lay] += std::abs(t.z());
	if (thke[lay] <= 0) thke[lay] = dz;
	bool found = false;
	for (unsigned int k=0; k<zxe.size(); k++) {
	  if (std::abs(zxe[k]-std::abs(t.z())) < 0.01) {
	    found = true;
	    break;
	  }
	}
	if (!found) {
	  zxe.push_back(std::abs(t.z()));
	  rhoxe.push_back(t.Rho()*std::cos(t.phi()));
	  dyxe.push_back(dy*std::cos(t.phi()));
	  dx1 -= 0.5*(t.rho()-dy)*std::cos(t.phi())*std::tan(10*CLHEP::deg);
	  dx2 -= 0.5*(t.rho()+dy)*std::cos(t.phi())*std::tan(10*CLHEP::deg);
	  dx1e.push_back(-dx1);
	  dx2e.push_back(-dx2);
	  laye.push_back(lay);
	}
      }
      if (copy[nsiz-1] == 21) {
	int iz = copy[nsiz-7];
	int fi = copy[nsiz-5];
	unsigned int it1 = find(iz, ize);
	if (it1 == ize.size())  ize.push_back(iz);
	unsigned int it2 = find(fi, phie);
	if (it2 == phie.size()) phie.push_back(fi);
      }
    } else if (idet == 5) {
      // HF
      if (!hf) {
	const std::vector<double> & paras = sol.parameters();
#ifdef DebugLog
	LogDebug("HCalGeom") << "HF " << sol.name() << " Shape " << sol.shape()
			     << " Z " << t.z() << " with " << paras.size()
			     << " Parameters";
	for (unsigned j=0; j<paras.size(); j++)
	  LogDebug("HCalGeom") << "HF Parameter[" << j << "] = " << paras[j];
#endif
	zf  = fabs(t.z());
	if (sol.shape() == ddpolycone_rrz) {
	  int nz  = (int)(paras.size())-3;
	  zf     += paras[3];
	  dzVcal  = 0.5*(paras[nz]-paras[3]);
	  hf      = true;
	} else if (sol.shape() == ddtubs || sol.shape() == ddcons) {
	  dzVcal  = paras[0];
	  zf     -= paras[0];
	  hf      = true;
	}
      }
#ifdef DebugLog
    } else {
      LogDebug("HCalGeom") << "Unknown Detector " << idet << " for " 
			   << sol.name() << " Shape " << sol.shape() << " R " 
			   << t.Rho() << " Z " << t.z();
#endif
    }
    dodet = fv.next();
  }

  int ibmx = 0, iemx = 0;
  for (int i = 0; i < 20; i++) {
    if (ib[i]>0) {
      rb[i] /= (double)(ib[i]);
      ibmx   = i+1;
    }
    if (ie[i]>0) {
      ze[i] /= (double)(ie[i]);
      iemx   = i+1;
    }
#ifdef DebugLog
    LogDebug("HCalGeom") << "Index " << i << " Barrel " << ib[i] << " "
			 << rb[i] << " Endcap " << ie[i] << " " << ze[i];
#endif
  }
  for (int i = 4; i >= 0; i--) {
    if (ib[i] == 0) {rb[i] = rb[i+1]; thkb[i] = thkb[i+1];}
    if (ie[i] == 0) {ze[i] = ze[i+1]; thke[i] = thke[i+1];}
#ifdef DebugLog
    if (ib[i] == 0 || ie[i] == 0)
      LogDebug("HCalGeom") << "Index " << i << " Barrel " << ib[i] << " "
			   << rb[i] << " Endcap " << ie[i] << " " << ze[i];
#endif
  }

#ifdef DebugLog
  for (unsigned int k=0; k<layb.size(); ++k)
    std::cout << "HB: " << layb[k] << " R " << rxb[k] << " " << rhoxb[k] << " Z " << zxb[k] << " DY " << dyxb[k] << " DZ " << dzxb[k] << "\n";
  for (unsigned int k=0; k<laye.size(); ++k) 
    std::cout << "HE: " << laye[k] << " R " << rhoxe[k] << " Z " << zxe[k] << " X1|X2 " << dx1e[k] << "|" << dx2e[k] << " DY " << dyxe[k] << "\n";

  printTile();
  LogDebug("HCalGeom") << "HcalNumberingFromDDD: Maximum Layer for HB " 
		       << ibmx << " for HE " << iemx << " Z for HF " << zf 
		       << " extent " << dzVcal;
#endif

  if (ibmx > 0) {
    rHB.resize(ibmx);
    drHB.resize(ibmx);
    for (int i=0; i<ibmx; i++) {
      rHB[i]  = rb[i];
      drHB[i] = thkb[i];
#ifdef DebugLog
      LogDebug("HCalGeom") << "HcalNumberingFromDDD: rHB[" << i << "] = "
			   << rHB[i] << " drHB[" << i << "] = " << drHB[i];
#endif
    }
  }
  if (iemx > 0) {
    zHE.resize(iemx);
    dzHE.resize(iemx);
    for (int i=0; i<iemx; i++) {
      zHE[i]  = ze[i];
      dzHE[i] = thke[i];
#ifdef DebugLog
      LogDebug("HCalGeom") << "HcalNumberingFromDDD: zHE[" << i << "] = "
			   << zHE[i] << " dzHE[" << i << "] = " << dzHE[i];
#endif
    }
  }

  nzHB   = (int)(izb.size());
  nmodHB = (int)(phib.size());
#ifdef DebugLog
  LogDebug("HCalGeom") << "HcalNumberingFromDDD::loadGeometry: " << nzHB
		       << " barrel half-sectors";
  for (int i=0; i<nzHB; i++)
    LogDebug("HCalGeom") << "Section " << i << " Copy number " << izb[i];
  LogDebug("HCalGeom") << "HcalNumberingFromDDD::loadGeometry: " << nmodHB
		       << " barrel modules";
  for (int i=0; i<nmodHB; i++)
    LogDebug("HCalGeom") << "Module " << i << " Copy number " << phib[i];
#endif

  nzHE   = (int)(ize.size());
  nmodHE = (int)(phie.size());
#ifdef DebugLog
  LogDebug("HCalGeom") << "HcalNumberingFromDDD::loadGeometry: " << nzHE
		       << " endcap half-sectors";
  for (int i=0; i<nzHE; i++)
    LogDebug("HCalGeom") << "Section " << i << " Copy number " << ize[i];
  LogDebug("HCalGeom") << "HcalNumberingFromDDD::loadGeometry: " << nmodHE
		       << " endcap modules";
  for (int i=0; i<nmodHE; i++)
    LogDebug("HCalGeom") << "Module " << i << " Copy number " << phie[i];
#endif

#ifdef DebugLog
  LogDebug("HCalGeom") << "HO has Z of size " << zho.size();
  for (unsigned int kk=0; kk<zho.size(); kk++)
    LogDebug("HCalGeom") << "ZHO[" << kk << "] = " << zho[kk];
#endif
  if (ibmx > 17 && zho.size() > 4) {
    rminHO   = rHB[17]-100.0;
    etaHO[0] = getEta(0.5*(rHB[17]+rHB[18]), zho[1]);
    etaHO[1] = getEta(rHB[18]+drHB[18], zho[2]);
    etaHO[2] = getEta(rHB[18]-drHB[18], zho[3]);
    etaHO[3] = getEta(rHB[18]+drHB[18], zho[4]);
  } else {
    rminHO   =-1.0;
    etaHO[0] = etaTable[4];
    etaHO[1] = etaTable[4];
    etaHO[2] = etaTable[10];
    etaHO[3] = etaTable[10];
  }
#ifdef DebugLog
  LogDebug("HCalGeom") << "HO Eta boundaries " << etaHO[0] << " " << etaHO[1]
		       << " " << etaHO[2] << " " << etaHO[3];
  std::cout << "HO Parameters " << rminHO << " " << zho.size();
  for (int i=0; i<4; ++i) std::cout << " eta[" << i << "] = " << etaHO[i];
  for (unsigned int i=0; i<zho.size(); ++i) std::cout << " zho[" << i << "] = " << zho[i];
  std::cout << std::endl;
#endif
}

std::vector<double> HcalNumberingFromDDD::getDDDArray(const std::string & str, 
						      const DDsvalues_type & sv,
						      int & nmin) const {
#ifdef DebugLog
  LogDebug("HCalGeom") << "HcalNumberingFromDDD:getDDDArray called for " 
		       << str << " with nMin "  << nmin;
#endif
  DDValue value(str);
  if (DDfetch(&sv,value)) {
#ifdef DebugLog
    LogDebug("HCalGeom") << "HcalNumberingFromDDD: " << value;
#endif
    const std::vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nmin > 0) {
      if (nval < nmin) {
	edm::LogError("HCalGeom") << "HcalNumberingFromDDD : # of " << str 
				  << " bins " << nval << " < " << nmin 
				  << " ==> illegal";
	throw cms::Exception("DDException") << "HcalNumberingFromDDD: cannot get array " << str;
      }
    } else {
      if (nval < 2) {
	edm::LogError("HCalGeom") << "HcalNumberingFromDDD : # of " << str
				  << " bins " << nval << " < 2 ==> illegal"
				  << " (nmin=" << nmin << ")";
	throw cms::Exception("DDException") << "HcalNumberingFromDDD: cannot get array " << str;
      }
    }
    nmin = nval;
    return fvec;
  } else {
    edm::LogError("HCalGeom") << "HcalNumberingFromDDD: cannot get array "
			      << str;
    throw cms::Exception("DDException") << "HcalNumberingFromDDD: cannot get array " << str;
  }
}

int HcalNumberingFromDDD::getShift(HcalSubdetector subdet, int depth) const {

  int shift;
  switch(subdet) {
  case HcalEndcap:
    shift = shiftHE[depth-1];
    break;
  case HcalForward:
    shift = shiftHF[depth-1];
    break;
  default:
    shift = shiftHB[depth-1];
    break;
  }
  return shift;
}

double HcalNumberingFromDDD::getGain(HcalSubdetector subdet, int depth) const {

  double gain;
  switch(subdet) {
  case HcalEndcap:
    gain = gainHE[depth-1];
    break;
  case HcalForward:
    gain = gainHF[depth-1];
    break;
  default:
    gain = gainHB[depth-1];
    break;
  }
  return gain;
}

unsigned int HcalNumberingFromDDD::find(int element, 
					std::vector<int>& array) const {

  unsigned int id = array.size();
  for (unsigned int i = 0; i < array.size(); i++) {
    if (element == array[i]) {
      id = i;
      break;
    }
  }
  return id;
}

int HcalNumberingFromDDD::unitPhi(int det, int etaR) const {

  const double fiveDegInRad = 2*M_PI/72;
  int units=0;
  if (det == static_cast<int>(HcalForward))
    units=int(phibin[nEta+etaR-etaMin[2]-1]/fiveDegInRad+0.5);
  else 
    units=int(phibin[etaR-1]/fiveDegInRad+0.5);

  return units;
}

void HcalNumberingFromDDD::tileHB(int eta, int depth) {

  double etaL   = etaTable[eta-1];
  double thetaL = 2.*atan(exp(-etaL));
  double etaH   = etaTable[eta];
  double thetaH = 2.*atan(exp(-etaH));
  int    layL=0, layH=0;
  if (depth == 1) {
    layH = depth1[eta-1];
  } else {
    layL = depth1[eta-1];
    layH = depth2[eta-1];
  }
  std::cout << "\ntileHB:: eta|depth " << eta << "|" << depth << " theta " << thetaH/CLHEP::deg << ":" << thetaL/CLHEP::deg << " Layer " << layL << ":" << layH-1 << "\n";
  for (int lay=layL; lay<layH; ++lay) {
    std::vector<double> area(2,0);
    int kk=0;
    for (unsigned int k=0; k<layb.size(); ++k) {
      if (lay == layb[k]) {
	double zmin = rhoxb[k]*std::cos(thetaL)/std::sin(thetaL);
	double zmax = rhoxb[k]*std::cos(thetaH)/std::sin(thetaH);
	double dz   = (std::min(zmax,dzxb[k]) - zmin);
	if (dz > 0) {
	  area[kk] = dz*dyxb[k];
	  kk++;
	}
      }
    }
    if (area[0] > 0) std::cout << std::setw(2) << lay << " Area " << std::setw(8) << area[0] << " " << std::setw(8) << area[1] << "\n";
  }
}

void HcalNumberingFromDDD::tileHE(int eta, int depth) {

  double etaL   = etaTable[eta-1];
  double thetaL = 2.*atan(exp(-etaL));
  double etaH   = etaTable[eta];
  double thetaH = 2.*atan(exp(-etaH));
  int    layL=0, layH=0;
  if (eta == 16) {
    layH = depth3[eta-1];
  } else if (depth == 1) {
    layH = depth1[eta-1];
  } else if (depth == 2) {
    layL = depth1[eta-1];
    layH = depth2[eta-1];
  } else {
    layL = depth2[eta-1];
    layH = depth3[eta-1];
  }
  double phib  = phibin[eta-1];
  int nphi = 2;
  if (phib > 6*CLHEP::deg) nphi = 1;
  std::cout << "\ntileHE:: Eta/depth " << eta << "|" << depth << " theta " << thetaH/CLHEP::deg << ":" << thetaL/CLHEP::deg << " Layer " << layL << ":" << layH-1 << " phi " << nphi << "\n";
  for (int lay=layL; lay<layH; ++lay) {
    std::vector<double> area(4,0);
    int kk=0;
    for (unsigned int k=0; k<laye.size(); ++k) {
      if (lay == laye[k]) {
	double rmin = zxe[k]*std::tan(thetaH);
	double rmax = zxe[k]*std::tan(thetaL);
	if ((lay != 0 || eta == 18) && 
	    (lay != 1 || (eta == 18 && rhoxe[k]-dyxe[k] > 1000) || (eta != 18 && rhoxe[k]-dyxe[k] < 1000)) &&
	    rmin+30 < rhoxe[k]+dyxe[k] && rmax > rhoxe[k]-dyxe[k]) {
	  rmin = std::max(rmin,rhoxe[k]-dyxe[k]);
	  rmax = std::min(rmax,rhoxe[k]+dyxe[k]);
	  double dx1 = rmin*std::tan(phib);
	  double dx2 = rmax*std::tan(phib);
	  double ar1=0, ar2=0;
	  if (nphi == 1) {
	    ar1 = 0.5*(rmax-rmin)*(dx1+dx2-4.*dx1e[k]);
	  } else {
	    ar1 = 0.5*(rmax-rmin)*(dx1+dx2-2.*dx1e[k]);
	    ar2 = 0.5*(rmax-rmin)*((rmax+rmin)*tan(10.*CLHEP::deg)-4*dx1e[k])-ar1;
	  }
	  area[kk] = ar1;
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

double HcalNumberingFromDDD::getEtaHO(double& etaR, double& x, double& y, 
				      double& z) const {

  double eta  = fabs(etaR);
  double r    = std::sqrt(x*x+y*y);
  if (r > rminHO) {
    double zz = fabs(z);
    if (zz > zho[3]) {
      if (eta <= etaTable[10]) eta = etaTable[10]+0.001;
    } else if (zz > zho[1]) {
      if (eta <=  etaTable[4]) eta = etaTable[4]+0.001;
    }
  }
  eta = (z >= 0. ? eta : -eta);
  //  std::cout << "R " << r << " Z " << z << " eta " << etaR << ":" << eta <<"\n";
  //  if (eta != etaR) std::cout << "**** Check *****\n";
  return eta;
}
