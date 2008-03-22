///////////////////////////////////////////////////////////////////////////////
// File: HcalNumberingFromDDD.cc
// Description: Usage of DDD to get to numbering scheme for hadron calorimeter
///////////////////////////////////////////////////////////////////////////////

#include "Geometry/HcalCommonData/interface/HcalNumberingFromDDD.h"

#include "DetectorDescription/Base/interface/DDException.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDValue.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"
#include <iostream>

HcalNumberingFromDDD::HcalNumberingFromDDD(std::string & name,
					   const DDCompactView & cpv) {
  edm::LogInfo("HCalGeom") << "Creating HcalNumberingFromDDD";
  initialize(name, cpv);
}

HcalNumberingFromDDD::~HcalNumberingFromDDD() {
  edm::LogInfo("HCalGeom") << "Deleting HcalNumberingFromDDD";
}

HcalNumberingFromDDD::HcalID HcalNumberingFromDDD::unitID(int det,
							  Hep3Vector point,
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
    } else {
      hsubdet = static_cast<int>(HcalEndcap);
    }
  }

  LogDebug("HCalGeom") << "HcalNumberingFromDDD: point = " << point << " det "
		       << hsubdet << " eta/R " << etaR << " phi " << hphi;

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
    } else {
      fioff   = phioff[1];
      if (ieta <= etaMin[1]) ieta = etaMin[1];
    }
    fibin = phibin[ieta-1];
  }

  int    nphi  = int((twopi+0.1*fibin)/fibin);
  int    zside = etaR>0 ? 1: 0;
  double hphi  = phi+fioff;
  if (hphi < 0)    hphi += twopi;
  int    iphi  = int(hphi/fibin) + 1;
  if (iphi > nphi) iphi = 1;

  LogDebug("HCalGeom") << "HcalNumberingFromDDD: etaR = " << etaR << " : "
		       << zside << "/" << ieta << " phi " << hphi << " : "
		       << iphi;
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

  const double fiveDegInRad = 2*M_PI/72;

  int iphi_skip=phi;
  int units=0;
  if (det==HcalForward) units=int(phibin[nEta+etaR-etaMin[2]-1]/fiveDegInRad+0.5);
  else units=int(phibin[etaR-1]/fiveDegInRad+0.5);

  if (units==2)      iphi_skip  = (phi-1)*2+1;
  else if (units==4) iphi_skip  = (phi-1)*4-1;
  if (iphi_skip < 0) iphi_skip += 72;

  LogDebug("HCalGeom") << "HcalNumberingFromDDD: phi units=" <<  units  
                       <<  "  iphi_skip=" << iphi_skip; 

  HcalNumberingFromDDD::HcalID tmp(det,zside,depth,etaR,phi,iphi_skip,lay);

  LogDebug("HCalGeom") << "HcalNumberingFromDDD: det = " << det << " " 
		       << tmp.subdet << " zside = " << tmp.zside << " depth = "
		       << tmp.depth << " eta/R = " << tmp.etaR << " phi = " 
		       << tmp.phi << " layer = " << tmp.lay;
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
	LogDebug("HCalGeom") << "HcalNumberingFromDDD: wrong eta " << etaR 
			     << " ("  << ir << "/" << nR << ") Detector "
			     << idet;
      }
      if (depth != 1 && depth != 2) {
	ok     = false;
	LogDebug("HCalGeom") << "HcalNumberingFromDDD: wrong depth " << depth
			     << " in Detector " << idet;
      }
    } else if (etaR <= nEta) {
      int depth0 = depth1[etaR-1];
      int kphi   = iphi + int((phioff[3]+0.1)/fibin);
      kphi       = (kphi-1)%4 + 1;
      if (etaR == nOff[0] && (kphi == 2 || kphi == 3)) depth0--;
      int laymin, laymax;
      if (depth == 1) {
	laymin = 1;
	laymax = depth0;
      } else if (depth == 2) {
	laymin = depth0+1;
        laymax = depth2[etaR-1];
	if (etaR==etaMax[0] && idet==static_cast<int>(HcalBarrel) &&
	    nOff.size()>3) laymax = nOff[3];
      } else  if (depth == 3) {
	laymin = depth2[etaR-1]+1;
        laymax = depth3[etaR-1];
	if (etaR<=etaMin[1] && idet==static_cast<int>(HcalEndcap)) {
	  if (nOff.size() > 4) laymin = nOff[4];
	  else                 laymin = 1;
	}
      } else {
	laymin = depth3[etaR-1]+1;
	laymax = maxlay;
      }
      if (laymin <= maxlay && laymax <= maxlay && laymin <= laymax) {
	if (idet == static_cast<int>(HcalEndcap)) {
	  flagrz = false;
	  rz     = 0.5*(zHE[laymax-1]+zHE[laymin-1]);
	  drz    = 0.5*(zHE[laymax-1]-zHE[laymin-1]);
	} else {
	  rz     = 0.5*(rHB[laymax-1]+rHB[laymin-1]);
	  drz    = 0.5*(rHB[laymax-1]-rHB[laymin-1]);
	}
      } else {
	ok = false;
	LogDebug("HCalGeom") << "HcalNumberingFromDDD: wrong depth " << depth
			     << " (Layer minimum " << laymin << " maximum " 
			     << laymax << " maxLay " << maxlay << ")";
      }
    } else {
      ok = false;
      LogDebug("HCalGeom") << "HcalNumberingFromDDD: wrong eta " << etaR
			   << "/" << nEta << " Detector " << idet;
    }
  } else {
    ok = false;
    LogDebug("HCalGeom") << "HcalNumberingFromDDD: wrong eta " << etaR 
			 << " det " << idet;
  }
  HcalCellType::HcalCell tmp(ok,eta,deta,phi,dphi,rz,drz,flagrz);

  LogDebug("HCalGeom") << "HcalNumberingFromDDD: det/side/depth/etaR/phi "
		       << det  << "/" << zside << "/" << depth << "/" << etaR
		       << "/" << iphi << " Cell Flag " << tmp.ok << " " 
		       << tmp.eta << " " << tmp.deta << " phi " << tmp.phi 
		       << " " << tmp.dphi << " r(z) " << tmp.rz  << " " 
		       << tmp.drz << " " << tmp.flagrz;
  return tmp;
}

std::vector<double> HcalNumberingFromDDD::getEtaTable() const {

  std::vector<double> tmp = etaTable;
  return tmp;
}

std::vector<HcalCellType::HcalCellType> HcalNumberingFromDDD::HcalCellTypes() const{

  std::vector<HcalCellType::HcalCellType> cellTypes =HcalCellTypes(HcalBarrel);
  LogDebug ("HCalGeom") << "HcalNumberingFromDDD: " << cellTypes.size()
			<< " cells of type HCal Barrel";
  for (unsigned int i=0; i<cellTypes.size(); i++)
    LogDebug ("HCalGeom") << "Cell " << i << " " << cellTypes[i];

  std::vector<HcalCellType::HcalCellType> hoCells   =HcalCellTypes(HcalOuter);
  LogDebug ("HCalGeom") << "HcalNumberingFromDDD: " << hoCells.size()
			<< " cells of type HCal Outer";
  for (unsigned int i=0; i<hoCells.size(); i++)
    LogDebug ("HCalGeom") << "Cell " << i << " " << hoCells[i];

  cellTypes.insert(cellTypes.end(), hoCells.begin(), hoCells.end());
  std::vector<HcalCellType::HcalCellType> heCells   =HcalCellTypes(HcalEndcap);
  LogDebug ("HCalGeom") << "HcalNumberingFromDDD: " << heCells.size()
			<< " cells of type HCal Endcap";
  for (unsigned int i=0; i<heCells.size(); i++)
    LogDebug ("HCalGeom") << "Cell " << i << " " << heCells[i];

  cellTypes.insert(cellTypes.end(), heCells.begin(), heCells.end());
  std::vector<HcalCellType::HcalCellType> hfCells   =HcalCellTypes(HcalForward);
  LogDebug ("HCalGeom") << "HcalNumberingFromDDD: " << hfCells.size()
			<< " cells of type HCal Forward";
  for (unsigned int i=0; i<hfCells.size(); i++)
    LogDebug ("HCalGeom") << "Cell " << i << " " << hfCells[i];

  cellTypes.insert(cellTypes.end(), hfCells.begin(), hfCells.end());
  return cellTypes;
}

std::vector<HcalCellType::HcalCellType> HcalNumberingFromDDD::HcalCellTypes(HcalSubdetector subdet) const {

  std::vector<HcalCellType::HcalCellType> cellTypes;
  if (subdet == HcalForward) {
    if (dzVcal < 0) return cellTypes;
  }

  int    dmin, dmax, indx, nz, nmod;
  switch(subdet) {
  case HcalEndcap:
    dmin = 1; dmax = 3; indx = 1, nz = nzHE, nmod = nmodHE;
    break;
  case HcalForward:
    dmin = 1; dmax = 2; indx = 2, nz = 2, nmod = 18;
    break;
  case HcalOuter:
    dmin = 4; dmax = 4; indx = 0, nz = nzHB, nmod = nmodHB;
    break;
  default:
    dmin = 1; dmax = 3; indx = 0, nz = nzHB, nmod = nmodHB;
    break;
  }

  int phi = 1, zside  = 1;
  bool cor = false;

  // Get the Cells 
  int subdet0 = static_cast<int>(subdet);
  for (int depth=dmin; depth<=dmax; depth++) {
    int    shift = getShift(subdet, depth);
    double gain  = getGain (subdet, depth);
    for (int eta=etaMin[indx]; eta<= etaMax[indx]; eta++) {
      HcalCellType::HcalCell temp1 = cell(subdet0, zside, depth, eta, phi,cor);
      if (temp1.ok) {
	HcalCellType::HcalCellType temp2(subdet, eta, phi, depth, temp1,
					 shift, gain, nz, nmod);
	cellTypes.push_back(temp2);
      }
    }
  }
  return cellTypes;
}

double HcalNumberingFromDDD::getEta(int det, int etaR, int zside,
				    int depth) const {

  double tmp = 0;
  if (det == static_cast<int>(HcalForward)) {
    int ir = nR + etaMin[2] - etaR - 1;
    if (ir > 0 && ir < nR) 
      tmp = 0.5*(getEta(rTable[ir-1],zVcal)+getEta(rTable[ir],zVcal));
  } else {
    if (etaR > 0 && etaR < nEta) {
      if (etaR == nOff[1]-1 && depth > 2) {
	tmp = 0.5*(etaTable[etaR+1]+etaTable[etaR-1]);
      } else {
	tmp = 0.5*(etaTable[etaR]+etaTable[etaR-1]);
      }
    }
  } 
  if (zside == 0) tmp = -tmp;
  LogDebug("HCalGeom") << "HcalNumberingFromDDD::getEta " << etaR << " " 
		       << zside << " " << depth << " ==> " << tmp;
  return tmp;
}
 
double HcalNumberingFromDDD::getEta(double r, double z) const {

  double tmp = 0;
  if (z != 0) tmp = -log(tan(0.5*atan(r/z)));
  LogDebug("HCalGeom") << "HcalNumberingFromDDD::getEta " << r << " " << z 
		       << " ==> " << tmp;
  return tmp;
}

double HcalNumberingFromDDD::deltaEta(int det, int etaR, int depth) const {

  double tmp = 0;
  if (det == static_cast<int>(HcalForward)) {
    int ir = nR + etaMin[2] - etaR - 1;
    if (ir > 0 && ir < nR) 
      tmp = 0.5*(getEta(rTable[ir-1],zVcal)-getEta(rTable[ir],zVcal));
  } else {
    if (etaR > 0 && etaR < nEta) {
      if (etaR == nOff[1]-1 && depth > 2) {
	tmp = 0.5*(etaTable[etaR+1]-etaTable[etaR-1]);
      } else {
	tmp = 0.5*(etaTable[etaR]-etaTable[etaR-1]);
      }
    } 
  }
  LogDebug("HCalGeom") << "HcalNumberingFromDDD::deltaEta " << etaR << " " 
		       << depth << " ==> " << tmp;
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
    throw DDException("HcalNumberingFromDDD: cannot match "+attribute+" to "+name);
  }

  std::vector<HcalCellType::HcalCellType> cellTypes =HcalCellTypes();
  LogDebug ("HCalGeom") << "HcalNumberingFromDDD: " << cellTypes.size()
			<< " cells of type HCal (All)";
  for (unsigned int i=0; i<cellTypes.size(); i++)
    LogDebug ("HCalGeom") << "Cell " << i << " " << cellTypes[i];

}

void HcalNumberingFromDDD::loadSpecPars(DDFilteredView fv) {

  DDsvalues_type sv(fv.mergedSpecifics());

  // Phi Offset
  int i, nphi=4;
  std::vector<double> tmp1 = getDDDArray("phioff",sv,nphi);
  phioff.resize(tmp1.size());
  for (i=0; i<nphi; i++) {
    phioff[i] = tmp1[i];
    LogDebug("HCalGeom") << "HcalNumberingFromDDD: phioff[" << i << "] = "
			 << phioff[i]/deg;
  }

  //Eta table
  nEta     = -1;
  std::vector<double> tmp2 = getDDDArray("etaTable",sv,nEta);
  etaTable.resize(tmp2.size());
  for (i=0; i<nEta; i++) {
    etaTable[i] = tmp2[i];
    LogDebug("HCalGeom") << "HcalNumberingFromDDD: etaTable[" << i << "] = "
			 << etaTable[i];
  }

  //R table
  nR     = -1;
  std::vector<double> tmp3 = getDDDArray("rTable",sv,nR);
  rTable.resize(tmp3.size());
  for (i=0; i<nR; i++) {
    rTable[i] = tmp3[i];
    LogDebug("HCalGeom") << "HcalNumberingFromDDD: rTable[" << i << "] = "
			 << rTable[i]/cm;
  }

  //Phi bins
  nPhi   = nEta + nR - 2;
  std::vector<double> tmp4 = getDDDArray("phibin",sv,nPhi);
  phibin.resize(tmp4.size());
  for (i=0; i<nPhi; i++) {
    phibin[i] = tmp4[i];
    LogDebug("HCalGeom") << "HcalNumberingFromDDD: phibin[" << i << "] = "
			 << phibin[i]/deg;
  }

  //Layer boundaries for depths 1, 2, 3, 4
  nDepth            = nEta - 1;
  std::vector<double> d1 = getDDDArray("depth1",sv,nDepth);
  nDepth            = nEta - 1;
  std::vector<double> d2 = getDDDArray("depth2",sv,nDepth);
  nDepth            = nEta - 1;
  std::vector<double> d3 = getDDDArray("depth3",sv,nDepth);
  LogDebug("HCalGeom") << "HcalNumberingFromDDD: " << nDepth << " Depths";
  depth1.resize(nDepth);
  depth2.resize(nDepth);
  depth3.resize(nDepth);
  for (i=0; i<nDepth; i++) {
    depth1[i] = static_cast<int>(d1[i]);
    depth2[i] = static_cast<int>(d2[i]);
    depth3[i] = static_cast<int>(d3[i]);
    LogDebug("HCalGeom") << "HcalNumberingFromDDD: depth1[" << i << "] = " 
			 << depth1[i] << " depth2[" << i << "]  = "<< depth2[i]
			 << " depth3[" << i << "] = " << depth3[i];
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
  for (i=0; i<ndx; i++) 
    LogDebug("HCalGeom") << "HcalNumberingFromDDD: etaMin[" << i << "] = "
			 << etaMin[i] << " etaMax[" << i << "] = "<< etaMax[i];

  // Geometry parameters for HF
  int ngpar = 7;
  std::vector<double> gpar = getDDDArray("gparHF",sv,ngpar);
  zVcal = gpar[6];
  LogDebug("HCalGeom") << "HcalNumberingFromDDD: zVcal " << zVcal;

  // nOff
  int noff = 3;
  std::vector<double>  nvec = getDDDArray("noff",sv,noff);
  nOff.resize(noff);
  for (i=0; i<noff; i++) {
    nOff[i] = static_cast<int>(nvec[i]);
    LogDebug("HCalGeom") << "HcalNumberingFromDDD: nOff[" << i << "] = " 
			 << nOff[i];
  }

  //Gains and Shifts for HB depths
  ndx                  = 4;
  gainHB               = getDDDArray("HBGains",sv,ndx);
  std::vector<double>  tmp7 = getDDDArray("HBShift",sv,ndx);
  shiftHB.resize(ndx);
  LogDebug("HCalGeom") << "HcalNumberingFromDDD:: Gain factor and Shift for "
		       << "HB depth layers:";
  for (i=0; i<ndx; i++) {
    shiftHB[i] = static_cast<int>(tmp7[i]);
    LogDebug("HCalGeom") <<"HcalNumberingFromDDD:: gainHB[" <<  i << "] = " 
			 << gainHB[i] << " shiftHB[" << i << "] = " 
			 << shiftHB[i];
  }

  //Gains and Shifts for HB depths
  ndx                  = 4;
  gainHE               = getDDDArray("HEGains",sv,ndx);
  std::vector<double>  tmp8 = getDDDArray("HEShift",sv,ndx);
  shiftHE.resize(ndx);
   LogDebug("HCalGeom") << "HcalNumberingFromDDD:: Gain factor and Shift for "
			<< "HE depth layers:";
  for (i=0; i<ndx; i++) {
    shiftHE[i] = static_cast<int>(tmp8[i]);
    LogDebug("HCalGeom") <<"HcalNumberingFromDDD:: gainHE[" <<  i << "] = " 
			 << gainHE[i] << " shiftHE[" << i << "] = " 
			 << shiftHE[i];
  }

  //Gains and Shifts for HF depths
  ndx                  = 4;
  gainHF               = getDDDArray("HFGains",sv,ndx);
  std::vector<double>  tmp9 = getDDDArray("HFShift",sv,ndx);
  shiftHF.resize(ndx);
  LogDebug("HCalGeom") << "HcalNumberingFromDDD:: Gain factor and Shift for "
		       << "HF depth layers:";
  for (i=0; i<ndx; i++) {
    shiftHF[i] = static_cast<int>(tmp9[i]);
    LogDebug("HCalGeom") <<"HcalNumberingFromDDD:: gainHF[" <<  i << "] = " 
			 << gainHF[i] << " shiftHF[" << i << "] = " 
			 << shiftHF[i];
  }
}

void HcalNumberingFromDDD::loadGeometry(DDFilteredView fv) {

  bool dodet=true, hf=false;
  std::vector<double> rb(20,0.0), ze(20,0.0);
  std::vector<int>    ib(20,0),   ie(20,0);
  std::vector<int>    izb, phib, ize, phie, izf, phif;
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
    if (idet == 3) {
      // HB
      LogDebug("HCalGeom") << "HB " << sol.name() << " Shape " << sol.shape()
			   << " Layer " << lay << " R " << t.Rho();
      if (lay >=0 && lay < 20) {
	ib[lay]++;
	rb[lay] += t.Rho();
      }
      if (lay == 0) {
	int iz = copy[nsiz-5];
	int fi = copy[nsiz-4];
	unsigned int it1 = find(iz, izb);
	if (it1 == izb.size())  izb.push_back(iz);
	unsigned int it2 = find(fi, phib);
	if (it2 == phib.size()) phib.push_back(fi);
      }
    } else if (idet == 4) {
      // HE
      LogDebug("HCalGeom") << "HE " << sol.name() << " Shape " << sol.shape()
			   << " Layer " << lay << " Z " << t.z();
      if (lay >=0 && lay < 20) {
	ie[lay]++;
	ze[lay] += fabs(t.z());
      }
      if (copy[nsiz-1] == 10) {
	int iz = copy[nsiz-6];
	int fi = copy[nsiz-4];
	unsigned int it1 = find(iz, ize);
	if (it1 == ize.size())  ize.push_back(iz);
	unsigned int it2 = find(fi, phie);
	if (it2 == phie.size()) phie.push_back(fi);
      }
    } else if (idet == 5) {
      // HF
      if (!hf) {
	const std::vector<double> & paras = sol.parameters();
	LogDebug("HCalGeom") << "HF " << sol.name() << " Shape " << sol.shape()
			     << " Z " << t.z() << " with " << paras.size()
			     << " Parameters";
	for (unsigned j=0; j<paras.size(); j++)
	  LogDebug("HCalGeom") << "HF Parameter[" << j << "] = " << paras[j];
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
    } else {
      LogDebug("HCalGeom") << "Unknown Detector " << idet << " for " 
			   << sol.name() << " Shape " << sol.shape() << " R " 
			   << t.Rho() << " Z " << t.z();
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
  }

  LogDebug("HCalGeom") << "HcalNumberingFromDDD: Maximum Layer for HB " 
		       << ibmx << " for HE " << iemx << " Z for HF " << zf 
		       << " extent " << dzVcal;
  if (ibmx > 0) {
    rHB.resize(ibmx);
    for (int i=0; i<ibmx; i++) {
      rHB[i] = rb[i];
      LogDebug("HCalGeom") << "HcalNumberingFromDDD: rHB[" << i << "] = "
			   << rHB[i];
    }
  }
  if (iemx > 0) {
    zHE.resize(iemx);
    for (int i=0; i<iemx; i++) {
      zHE[i] = ze[i];
      LogDebug("HCalGeom") << "HcalNumberingFromDDD: zHE[" << i << "] = "
			   << zHE[i];
    }
  }

  nzHB   = (int)(izb.size());
  nmodHB = (int)(phib.size());
  LogDebug("HCalGeom") << "HcalNumberingFromDDD::loadGeometry: " << nzHB
		       << " barrel half-sectors";
  for (int i=0; i<nzHB; i++)
    LogDebug("HCalGeom") << "Section " << i << " Copy number " << izb[i];
  LogDebug("HCalGeom") << "HcalNumberingFromDDD::loadGeometry: " << nmodHB
		       << " barrel modules";
  for (int i=0; i<nmodHB; i++)
    LogDebug("HCalGeom") << "Module " << i << " Copy number " << phib[i];

  nzHE   = (int)(ize.size());
  nmodHE = (int)(phie.size());
  LogDebug("HCalGeom") << "HcalNumberingFromDDD::loadGeometry: " << nzHE
		       << " endcap half-sectors";
  for (int i=0; i<nzHE; i++)
    LogDebug("HCalGeom") << "Section " << i << " Copy number " << ize[i];
  LogDebug("HCalGeom") << "HcalNumberingFromDDD::loadGeometry: " << nmodHE
		       << " endcap modules";
  for (int i=0; i<nmodHE; i++)
    LogDebug("HCalGeom") << "Module " << i << " Copy number " << phie[i];

}

std::vector<double> HcalNumberingFromDDD::getDDDArray(const std::string & str, 
						      const DDsvalues_type & sv,
						      int & nmin) const {
  LogDebug("HCalGeom") << "HcalNumberingFromDDD:getDDDArray called for " 
		       << str << " with nMin "  << nmin;
  DDValue value(str);
  if (DDfetch(&sv,value)) {
    LogDebug("HCalGeom") << "HcalNumberingFromDDD: " << value;
    const std::vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nmin > 0) {
      if (nval < nmin) {
	edm::LogError("HCalGeom") << "HcalNumberingFromDDD : # of " << str 
				  << " bins " << nval << " < " << nmin 
				  << " ==> illegal";
	throw DDException("HcalNumberingFromDDD: cannot get array "+str);
      }
    } else {
      if (nval < 2) {
	edm::LogError("HCalGeom") << "HcalNumberingFromDDD : # of " << str
				  << " bins " << nval << " < 2 ==> illegal"
				  << " (nmin=" << nmin << ")";
	throw DDException("HcalNumberingFromDDD: cannot get array "+str);
      }
    }
    nmin = nval;
    return fvec;
  } else {
    edm::LogError("HCalGeom") << "HcalNumberingFromDDD: cannot get array "
			      << str;
    throw DDException("HcalNumberingFromDDD: cannot get array "+str);
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
					std::vector<int> array) const {

  unsigned int id = array.size();
  for (unsigned int i = 0; i < array.size(); i++) {
    if (element == array[i]) {
      id = i;
      break;
    }
  }
  return id;
}
