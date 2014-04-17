#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

//#define DebugLog

HcalDDDSimConstants::HcalDDDSimConstants() : tobeInitialized(true) {

#ifdef DebugLog
  edm::LogInfo("HCalGeom") << "HcalDDDSimConstants::HcalDDDSimConstants constructor";
#endif

}

HcalDDDSimConstants::HcalDDDSimConstants(const DDCompactView& cpv) : tobeInitialized(true) {

#ifdef DebugLog
  edm::LogInfo("HCalGeom") << "HcalDDDSimConstants::HcalDDDSimConstants ( const DDCompactView& cpv ) constructor";
#endif

  initialize(cpv);

#ifdef DebugLog
  std::vector<HcalCellType> cellTypes = HcalCellTypes();
  edm::LogInfo ("HCalGeom") << "HcalDDDSimConstants: " << cellTypes.size()
			    << " cells of type HCal (All)";
#endif
}


HcalDDDSimConstants::~HcalDDDSimConstants() { 
#ifdef DebugLog
  std::cout << "destructed!!!" << std::endl;
#endif
}

HcalCellType::HcalCell HcalDDDSimConstants::cell(int idet, int zside, 
						 int depth, int etaR,
						 int iphi) const {

  checkInitialized();
  double etaMn = etaMin[0];
  double etaMx = etaMax[0];
  if (idet==static_cast<int>(HcalEndcap)) {
    etaMn = etaMin[1]; etaMx = etaMax[1];
  } else if (idet==static_cast<int>(HcalForward)) {
    etaMn = etaMin[2]; etaMx = etaMax[2];
  }
  double eta = 0, deta = 0, phi = 0, dphi = 0, rz = 0, drz = 0;
  bool   ok = false, flagrz = true;
  if ((idet==static_cast<int>(HcalBarrel)||idet==static_cast<int>(HcalEndcap)||
       idet==static_cast<int>(HcalOuter)||idet==static_cast<int>(HcalForward))
      && etaR >=etaMn && etaR <= etaMx && depth > 0)    ok = true;
  if (idet == static_cast<int>(HcalEndcap) && depth>(int)(zHE.size()))ok=false;
  else if (idet == static_cast<int>(HcalBarrel) && depth > 17)        ok=false;
  else if (idet == static_cast<int>(HcalOuter) && depth != 4)         ok=false;
  else if (idet == static_cast<int>(HcalForward) && depth > 2)        ok=false;
  if (ok) {
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
      fibin = phitable[etaR-etaMin[2]];
      if (unitPhi(fibin) > 2) fioff = phioff[4];
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
	edm::LogInfo("HCalGeom") << "HcalDDDSimConstants: wrong eta " << etaR 
			     << " ("  << ir << "/" << nR << ") Detector "
			     << idet;
#endif
      }
    } else if (etaR <= nEta) {
      int laymin(depth), laymax(depth);
      if (idet == static_cast<int>(HcalOuter)) {
	laymin = (etaR > nOff[2]) ? ((int)(zHE.size())) : ((int)(zHE.size()))-1;
	laymax = ((int)(zHE.size()));
      }
      double d1=0, d2=0;
      if (idet == static_cast<int>(HcalEndcap)) {
	flagrz = false;
	d1     = zHE[laymin-1] - dzHE[laymin-1];
	d2     = zHE[laymax-1] + dzHE[laymax-1];
      } else {
	d1     = rHB[laymin-1] - drHB[laymin-1];
	d2     = rHB[laymax-1] + drHB[laymax-1];
      }
      rz     = 0.5*(d2+d1);
      drz    = 0.5*(d2-d1);
    } else {
      ok = false;
#ifdef DebugLog
      edm::LogInfo("HCalGeom") << "HcalDDDSimConstants: wrong depth " << depth
			       << " or etaR " << etaR << " for detector " 
			       << idet;
#endif
    }
  } else {
    ok = false;
#ifdef DebugLog
    edm::LogInfo("HCalGeom") << "HcalDDDSimConstants: wrong depth " << depth
			     << " det " << idet;
#endif
  }
  HcalCellType::HcalCell tmp(ok,eta,deta,phi,dphi,rz,drz,flagrz);

#ifdef DebugLog
  edm::LogInfo("HCalGeom") << "HcalDDDSimConstants: det/side/depth/etaR/phi "
			   << idet  << "/" << zside << "/" << depth << "/" 
			   << etaR << "/" << iphi << " Cell Flag " << tmp.ok 
			   << " "  << tmp.eta << " " << tmp.deta << " phi " 
			   << tmp.phi << " " << tmp.dphi << " r(z) " << tmp.rz
			   << " "  << tmp.drz << " " << tmp.flagrz;
#endif
  return tmp;
}

std::vector<std::pair<double,double> > HcalDDDSimConstants::getConstHBHE(const int type) const {

  checkInitialized();
  std::vector<std::pair<double,double> > gcons;
  if (type == 0) {
    for (unsigned int i=0; i<rHB.size(); ++i) {
      gcons.push_back(std::pair<double,double>(rHB[i],drHB[i]));
    }
  } else {
    for (unsigned int i=0; i<zHE.size(); ++i) {
      gcons.push_back(std::pair<double,double>(zHE[i],dzHE[i]));
    }
  }
  return gcons;
}


std::pair<int,double> HcalDDDSimConstants::getDetEta(double eta, int depth) {

  checkInitialized();
  int    hsubdet(0), ieta(0);
  double etaR(0);
  double heta = fabs(eta);
  for (int i = 0; i < nEta; i++)
    if (heta > etaTable[i]) ieta = i + 1;
  if (heta <= etaRange[1]) {
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
  return std::pair<int,double>(hsubdet,etaR);
}

int HcalDDDSimConstants::getEta(int det,int lay, double hetaR) {

  checkInitialized();
  int    ieta(0);
  if (det == static_cast<int>(HcalForward)) { // Forward HCal
    ieta    = etaMax[2];
    for (int i = nR-1; i > 0; i--)
      if (hetaR < rTable[i]) ieta = etaMin[2] + nR - i - 1;
  } else { // Barrel or Endcap
    ieta  = 1;
    for (int i = 0; i < nEta-1; i++)
      if (hetaR > etaTable[i]) ieta = i + 1;
    if (det == static_cast<int>(HcalBarrel)) {
      if (ieta > etaMax[0])  ieta = etaMax[0];
      if (lay == 18) {
	if (hetaR > etaHO[1] && ieta == nOff[2]) ieta++;
      }
    } else {
      if (ieta <= etaMin[1]) ieta = etaMin[1];
    }
  }
  return ieta;
}

std::pair<int,int> HcalDDDSimConstants::getEtaDepth(int det, int etaR, int phi,
						    int depth, int lay) {

  checkInitialized();
  //Modify the depth index
  if (det == static_cast<int>(HcalForward)) { // Forward HCal
  } else if (det == static_cast<int>(HcalOuter)) {
    depth = 4;
  } else {
    if (lay >= 0) {
      depth= layerGroup[etaR-1][lay-1];
      if (etaR == nOff[0] && lay > 1) {
	int   kphi   = phi + int((phioff[3]+0.1)/phibin[etaR-1]);
	kphi         = (kphi-1)%4 + 1;
	if (kphi == 2 || kphi == 3) depth = layerGroup[etaR-1][lay-2];
      }
    } else if (det == static_cast<int>(HcalBarrel)) {
      if (depth==3) depth = 2;
    }
    if (etaR == nOff[1] && depth > 2) {
      etaR = nOff[1]-1;
    } else if (etaR == etaMin[1]) {
      if (det == static_cast<int>(HcalBarrel)) {
	if (depth > 2) depth = 2;
      } else {
	if (depth < 3) depth = 3;
      }
    }
  }
  return std::pair<int,int>(etaR,depth);
}

double HcalDDDSimConstants::getEtaHO(double& etaR, double& x, double& y, 
				     double& z) const {

  checkInitialized();
  if (zho.size() > 4) {
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
#ifdef DebugLog
    std::cout << "R " << r << " Z " << z << " eta " << etaR <<":" <<eta <<"\n";
    if (eta != etaR) std::cout << "**** Check *****\n";
#endif
    return eta;
  } else {
    return etaR;
  }
}

std::vector<double> HcalDDDSimConstants::getEtaTableHF() const {

  checkInitialized();
  std::vector<double> etas;
  for (unsigned int i=0; i<rTable.size(); ++i) {
    unsigned int k = rTable.size()-i-1;
    double eta = -log(tan(0.5*atan(rTable[k]/zVcal)));
    etas.push_back(eta);
  }
  return etas;
}

std::pair<int,int> HcalDDDSimConstants::getModHalfHBHE(const int type) const {

  checkInitialized();
  if (type == 0) {
    return std::pair<int,int>(nmodHB,nzHB);
  } else {
    return std::pair<int,int>(nmodHE,nzHE);
  }
}

std::pair<double,double> HcalDDDSimConstants::getPhiCons(int det, int ieta) {

  checkInitialized();
  double fioff(0), fibin(0);
  if (det == static_cast<int>(HcalForward)) { // Forward HCal
    fioff   = phioff[2];
    fibin   = phitable[ieta-etaMin[2]];
    if  (unitPhi(fibin) > 2) {   // HF double-phi  
      fioff = phioff[4];
    }
  } else { // Barrel or Endcap
    if (det == static_cast<int>(HcalBarrel)) {
      fioff   = phioff[0];
    } else {
      fioff   = phioff[1];
    }
    fibin = phibin[ieta-1];
  }
  return std::pair<double,double>(fioff,fibin);
}

std::vector<HcalCellType> HcalDDDSimConstants::HcalCellTypes() const{

  std::vector<HcalCellType> cellTypes =HcalCellTypes(HcalBarrel);
#ifdef DebugLog
  edm::LogInfo ("HCalGeom") << "HcalDDDSimConstants: " << cellTypes.size()
			<< " cells of type HCal Barrel";
  for (unsigned int i=0; i<cellTypes.size(); i++)
    edm::LogInfo ("HCalGeom") << "Cell " << i << " " << cellTypes[i];
#endif

  std::vector<HcalCellType> hoCells   =HcalCellTypes(HcalOuter);
#ifdef DebugLog
  edm::LogInfo ("HCalGeom") << "HcalDDDSimConstants: " << hoCells.size()
			<< " cells of type HCal Outer";
  for (unsigned int i=0; i<hoCells.size(); i++)
    edm::LogInfo ("HCalGeom") << "Cell " << i << " " << hoCells[i];
#endif
  cellTypes.insert(cellTypes.end(), hoCells.begin(), hoCells.end());

  std::vector<HcalCellType> heCells   =HcalCellTypes(HcalEndcap);
#ifdef DebugLog
  edm::LogInfo ("HCalGeom") << "HcalDDDSimConstants: " << heCells.size()
			<< " cells of type HCal Endcap";
  for (unsigned int i=0; i<heCells.size(); i++)
    edm::LogInfo ("HCalGeom") << "Cell " << i << " " << heCells[i];
#endif
  cellTypes.insert(cellTypes.end(), heCells.begin(), heCells.end());

  std::vector<HcalCellType> hfCells   =HcalCellTypes(HcalForward);
#ifdef DebugLog
  edm::LogInfo ("HCalGeom") << "HcalDDDSimConstants: " << hfCells.size()
			<< " cells of type HCal Forward";
  for (unsigned int i=0; i<hfCells.size(); i++)
    edm::LogInfo ("HCalGeom") << "Cell " << i << " " << hfCells[i];
#endif
  cellTypes.insert(cellTypes.end(), hfCells.begin(), hfCells.end());

  return cellTypes;
}

std::vector<HcalCellType> HcalDDDSimConstants::HcalCellTypes(HcalSubdetector subdet,
							     int ieta, int depthl) const {

  checkInitialized();
  std::vector<HcalCellType> cellTypes;
  if (subdet == HcalForward) {
    if (dzVcal < 0) return cellTypes;
  }

  int    dmin, dmax, indx, nz, nmod;
  double hsize = 0;
  switch(subdet) {
  case HcalEndcap:
    dmin = 1; dmax = 19; indx = 1; nz = nzHE; nmod = nmodHE;
    break;
  case HcalForward:
    dmin = 1; dmax = 2; indx = 2; nz = 2; nmod = 18; 
    break;
  case HcalOuter:
    dmin = 4; dmax = 4; indx = 0; nz = nzHB; nmod = nmodHB;
    break;
  default:
    dmin = 1; dmax = 17; indx = 0; nz = nzHB; nmod = nmodHB;
    break;
  }
  if (depthl > 0) dmin = dmax = depthl;
  int ietamin = (ieta>0) ? ieta : etaMin[indx];
  int ietamax = (ieta>0) ? ieta : etaMax[indx];

  int phi = 1, zside  = 1;

  // Get the Cells 
  int subdet0 = static_cast<int>(subdet);
  for (int depth=dmin; depth<=dmax; depth++) {
    int    shift = getShift(subdet, depth);
    double gain  = getGain (subdet, depth);
    if (subdet == HcalForward) {
      if (depth == 1) hsize = dzVcal;
      else            hsize = dzVcal-0.5*dlShort;
    }
    for (int eta=ietamin; eta<= ietamax; eta++) {
      HcalCellType::HcalCell temp1 = cell(subdet0,zside,depth,eta,phi);
      if (temp1.ok) {
	int units = unitPhi (subdet0, eta);
	HcalCellType temp2(subdet, eta, phi, depth, temp1,
			   shift, gain, nz, nmod, hsize, units);
	if (subdet == HcalOuter) {
	  if (eta == nOff[4]) {
	    std::vector<int> missPlus, missMinus;
	    int kk = 7;
	    for (int miss=0; miss<nOff[5]; miss++) {
	      missPlus.push_back(nOff[kk]);
	      kk++;
	    }
	    for (int miss=0; miss<nOff[6]; miss++) {
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

void HcalDDDSimConstants::initialize(const DDCompactView& cpv) {

  if (tobeInitialized) {
    tobeInitialized = false;

    std::string attribute = "OnlyForHcalSimNumbering"; 
    std::string value     = "any";
    DDValue val(attribute, value, 0.0);
  
    DDSpecificsFilter filter;
    filter.setCriteria(val, DDSpecificsFilter::not_equals,
		       DDSpecificsFilter::AND, true, // compare strings otherwise doubles
		       true  // use merged-specifics or simple-specifics
		       );
    DDFilteredView fv(cpv);
    fv.addFilter(filter);
    bool ok = fv.firstChild();

    if (ok) {
      //Load the SpecPars
      loadSpecPars(fv);

      //Load the Geometry parameters
      loadGeometry(fv);
    } else {
      edm::LogError("HCalGeom") << "HcalDDDSimConstants: cannot get filtered "
				<< " view for " << attribute << " not matching "
				<< value;
      throw cms::Exception("DDException") << "HcalDDDSimConstants: cannot match " << attribute << " to " << value;
    }
  }
}

unsigned int HcalDDDSimConstants::numberOfCells(HcalSubdetector subdet) const{

  unsigned int num = 0;
  std::vector<HcalCellType> cellTypes = HcalCellTypes(subdet);
  for (unsigned int i=0; i<cellTypes.size(); i++) {
    num += (unsigned int)(cellTypes[i].nPhiBins());
    if (cellTypes[i].nHalves() > 1) 
      num += (unsigned int)(cellTypes[i].nPhiBins());
    num -= (unsigned int)(cellTypes[i].nPhiMissingBins());
  }
#ifdef DebugLog
  edm::LogInfo ("HCalGeom") << "HcalDDDSimConstants:numberOfCells " 
			<< cellTypes.size()  << " " << num 
			<< " for subdetector " << subdet;
#endif
  return num;
}

int HcalDDDSimConstants::phiNumber(int phi, int units) const {

  int iphi_skip = phi;
  if      (units==2) iphi_skip  = (phi-1)*2+1;
  else if (units==4) iphi_skip  = (phi-1)*4-1;
  if (iphi_skip < 0) iphi_skip += 72;
  return iphi_skip;
}

void HcalDDDSimConstants::printTiles() const {
 
  checkInitialized();
  std::cout << "Tile Information for HB:\n" << "========================\n\n";
  for (int eta=etaMin[0]; eta<= etaMax[0]; eta++) {
    int dmax = 1;
    if (depths[0][eta-1] < 17) dmax = 2;
    for (int depth=1; depth<=dmax; depth++) 
      printTileHB(eta, depth);
  }

  std::cout << "\nTile Information for HE:\n" <<"========================\n\n";
  for (int eta=etaMin[1]; eta<= etaMax[1]; eta++) {
    int dmin=1, dmax=3;
    if (eta == etaMin[1]) {
      dmin = 3;
    } else if (depths[0][eta-1] > 18) {
      dmax = 1;
    } else if (depths[1][eta-1] > 18) {
      dmax = 2;
    }
    for (int depth=dmin; depth<=dmax; depth++)
      printTileHE(eta, depth);
  }
}

int HcalDDDSimConstants::unitPhi(int det, int etaR) const {

  checkInitialized();
  double dphi = (det == static_cast<int>(HcalForward)) ? phitable[etaR-etaMin[2]] : phibin[etaR-1];
  return unitPhi(dphi);
}

int HcalDDDSimConstants::unitPhi(double dphi) const {

  const double fiveDegInRad = 2*M_PI/72;
  int units = int(dphi/fiveDegInRad+0.5);
  return units;
}

void HcalDDDSimConstants::checkInitialized() const {
  if (tobeInitialized) {
    edm::LogError("HcalGeom") << "HcalDDDSimConstants : to be initialized correctly";
    throw cms::Exception("DDException") << "HcalDDDSimConstants: to be initialized";
  }
} 

void HcalDDDSimConstants::loadSpecPars(const DDFilteredView& fv) {

  DDsvalues_type sv(fv.mergedSpecifics());

  // Phi Offset
  int nphi=5;
  phioff = getDDDArray("phioff",sv,nphi);
#ifdef DebugLog
  std::cout << "HcalDDDSimConstants: " << nphi << " phioff values";
  for (int i=0; i<nphi; i++) 
    std::cout << " [" << i << "] = " << phioff[i]/CLHEP::deg;
  std::cout << std::endl;
#endif

  //Eta table
  nEta     = 0;
  etaTable = getDDDArray("etaTable",sv,nEta);
#ifdef DebugLog
  std::cout << "HcalDDDSimConstants: " << nEta << " entries for etaTable";
  for (int i=0; i<nEta; i++) std::cout << " [" << i << "] = " << etaTable[i];
  std::cout << std::endl;
#endif

  //R table
  nR     = 0;
  rTable = getDDDArray("rTable",sv,nR);
#ifdef DebugLog
  std::cout << "HcalDDDSimConstants: " << nR << " entries for rTable";
  for (int i=0; i<nR; i++) 
    std::cout << " [" << i << "] = " << rTable[i]/CLHEP::cm;
  std::cout << std::endl;
#endif

  //Phi bins
  nPhi   = nEta - 1;
  phibin = getDDDArray("phibin",sv,nPhi);
#ifdef DebugLog
  std::cout << "HcalDDDSimConstants: " << nPhi << " entries for phibin";
  for (int i=0; i<nPhi; i++)
    std::cout << " [" << i << "] = " << phibin[i]/CLHEP::deg;
  std::cout << std::endl;
#endif
  nPhiF = nR - 1;
  phitable = getDDDArray("phitable",sv,nPhiF);
#ifdef DebugLog
  std::cout << "HcalDDDSimConstants: " << nPhiF << " entries for phitable";
  for (int i=0; i<nPhiF; i++)
    std::cout << " [" << i << "] = " << phitable[i]/CLHEP::deg;
  std::cout << std::endl;
#endif

  //Layer grouping
  char name[20];
  int  layers = 19;
  for (int i=0; i<nEta-1; ++i) {
    sprintf (name, "layerGroupEta%d", i+1);
    layerGroup[i] = dbl_to_int(getDDDArray(name,sv,layers));
    if (layers == 0) {
      layerGroup[i] = layerGroup[i-1]; 
      layers        = (int)(layerGroup[i].size());
    }
#ifdef DebugLog
    std::cout << "HcalDDDSimConstants:Read " << name << ":";
    for (int k=0; k<layers; k++) 
      std::cout << " [" << k << "] = " << layerGroup[i][k];
    std::cout << std::endl;
#endif
    layers = -1;
  }

  // Minimum and maximum eta boundaries
  int ndx  = 3;
  etaMin   = dbl_to_int(getDDDArray("etaMin",sv,ndx));
  etaMax   = dbl_to_int(getDDDArray("etaMax",sv,ndx));
  etaRange = getDDDArray("etaRange",sv,ndx);
  etaMin[0] = 1;
  etaMax[1] = nEta-1;
  etaMax[2] = etaMin[2]+nR-2;
#ifdef DebugLog
  for (int i=0; i<ndx; i++) 
    std::cout << "HcalDDDSimConstants: etaMin[" << i << "] = " << etaMin[i]
	      << " etaMax[" << i << "] = "<< etaMax[i] << " etaRange[" << i 
	      << "] = " << etaRange[i] << std::endl;
#endif

  // Geometry parameters for HF
  int ngpar = 7;
  gparHF    = getDDDArray("gparHF",sv,ngpar);
  dlShort   = gparHF[0];
  zVcal     = gparHF[4];
#ifdef DebugLog
  std::cout << "HcalDDDSimConstants: dlShort " << dlShort << " zVcal " << zVcal
	    << " and " << ngpar << " other parameters";
  for (int i=0; i<ngpar; ++i)
    std::cout << " [" << i << "] = " << gparHF[i];
  std::cout << std::endl;
#endif

  // nOff
  int noff = 3;
  nOff     = dbl_to_int(getDDDArray("noff",sv,noff));
#ifdef DebugLog
  std::cout << "HcalDDDSimConstants: " << noff << " nOff parameters: ";
  for (int i=0; i<noff; i++)
    std::cout << " [" << i << "] = " << nOff[i];
  std::cout << std::endl;
#endif

  //Gains and Shifts for HB depths
  ndx      = 4;
  gainHB   = getDDDArray("HBGains",sv,ndx);
  shiftHB  = dbl_to_int(getDDDArray("HBShift",sv,ndx));
#ifdef DebugLog
  std::cout << "HcalDDDSimConstants:: Gain factor and Shift for HB depth "
	    << "layers:" << std::endl;
  for (int i=0; i<ndx; i++)
    std::cout <<" gainHB[" <<  i << "] = " << gainHB[i] << " shiftHB[" << i 
	      << "] = " << shiftHB[i];
  std::cout << std::endl;
#endif

  //Gains and Shifts for HE depths
  ndx      = 4;
  gainHE   = getDDDArray("HEGains",sv,ndx);
  shiftHE  = dbl_to_int(getDDDArray("HEShift",sv,ndx));
#ifdef DebugLog
  std::cout << "HcalDDDSimConstants:: Gain factor and Shift for HE depth "
	    << "layers:" << std::endl;
  for (int i=0; i<ndx; i++)
    std::cout <<" gainHE[" <<  i << "] = " << gainHE[i] << " shiftHE[" << i 
	      << "] = " << shiftHE[i];
  std::cout << std::endl;
#endif
  
  //Gains and Shifts for HF depths
  ndx      = 4;
  gainHF   = getDDDArray("HFGains",sv,ndx);
  shiftHF  = dbl_to_int(getDDDArray("HFShift",sv,ndx));
#ifdef DebugLog
  std::cout << "HcalDDDSimConstants:: Gain factor and Shift for HF depth "
	    << "layers:" << std::endl;
  for (int i=0; i<ndx; i++)
    std::cout <<" gainHF[" <<  i << "] = " << gainHF[i] << " shiftHF[" << i
	      << "] = " << shiftHF[i];
  std::cout << std::endl;
#endif

  //Transform some of the parameters
  maxDepth[0] = maxDepth[1] = 0;
  maxDepth[2] = maxDepth[3] = 4;
  for (int i=0; i<nEta-1; ++i) {
    unsigned int imx = layerGroup[i].size();
    int laymax = (imx > 0) ? layerGroup[i][imx-1] : 0;
    if (i < etaMax[0]) {
      int laymax0 = (imx > 16) ? layerGroup[i][16] : laymax;
      if (i+1 == etaMax[0] && laymax0 > 2) laymax0 = 2;
      if (maxDepth[0] < laymax0) maxDepth[0] = laymax0;
    }
    if (i >= etaMin[1]-1 && i < etaMax[1]) {
      if (maxDepth[1] < laymax) maxDepth[1] = laymax;
    }
  }
#ifdef DebugLog
  for (int i=0; i<4; ++i)
    std::cout << "Detector Type [" << i << "] iEta " << etaMin[i] << ":" 
	      << etaMax[i] << " MaxDepth " << maxDepth[i] << std::endl;
#endif

  int maxdepth = (maxDepth[1]>maxDepth[0]) ? maxDepth[1] : maxDepth[0];
  for (int i=0; i<maxdepth; ++i) {
    for (int k=0; k<nEta-1; ++k) {
      int layermx = ((k+1 < etaMin[1]) && i < maxDepth[0]) ? 17 : (int)layerGroup[k].size();
      int ll      = layermx;
      for (int l=layermx-1; l >= 0; --l) {
	if (layerGroup[k][l] == i+1) {
	  ll = l+1; break;
	}
      }
      depths[i].push_back(ll);
    }
#ifdef DebugLog
    std::cout << "Depth " << i << " with " << depths[i].size() << " etas:";
    for (int k=0; k<nEta-1; ++k) std::cout << " " << depths[i][k];
    std::cout << std::endl;
#endif
  }
}

void HcalDDDSimConstants::loadGeometry(const DDFilteredView& _fv) {

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
      edm::LogInfo("HCalGeom") << "HB " << sol.name() << " Shape " << sol.shape()
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
	  edm::LogInfo("HCalGeom") << "Detector " << idet << " Lay " << lay << " fi " << ifi << " " << ich << " z " << z1 << " " << z2;
#endif
	}
      }
    } else if (idet == 4) {
      // HE
#ifdef DebugLog
      edm::LogInfo("HCalGeom") << "HE " << sol.name() << " Shape " << sol.shape()
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
	edm::LogInfo("HCalGeom") << "HF " << sol.name() << " Shape " << sol.shape()
			     << " Z " << t.z() << " with " << paras.size()
			     << " Parameters";
	for (unsigned j=0; j<paras.size(); j++)
	  edm::LogInfo("HCalGeom") << "HF Parameter[" << j << "] = " << paras[j];
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
      edm::LogInfo("HCalGeom") << "Unknown Detector " << idet << " for " 
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
    edm::LogInfo("HCalGeom") << "Index " << i << " Barrel " << ib[i] << " "
			 << rb[i] << " Endcap " << ie[i] << " " << ze[i];
#endif
  }
  for (int i = 4; i >= 0; i--) {
    if (ib[i] == 0) {rb[i] = rb[i+1]; thkb[i] = thkb[i+1];}
    if (ie[i] == 0) {ze[i] = ze[i+1]; thke[i] = thke[i+1];}
#ifdef DebugLog
    if (ib[i] == 0 || ie[i] == 0)
      edm::LogInfo("HCalGeom") << "Index " << i << " Barrel " << ib[i] << " "
			   << rb[i] << " Endcap " << ie[i] << " " << ze[i];
#endif
  }

#ifdef DebugLog
  for (unsigned int k=0; k<layb.size(); ++k)
    std::cout << "HB: " << layb[k] << " R " << rxb[k] << " " << rhoxb[k] << " Z " << zxb[k] << " DY " << dyxb[k] << " DZ " << dzxb[k] << "\n";
  for (unsigned int k=0; k<laye.size(); ++k) 
    std::cout << "HE: " << laye[k] << " R " << rhoxe[k] << " Z " << zxe[k] << " X1|X2 " << dx1e[k] << "|" << dx2e[k] << " DY " << dyxe[k] << "\n";

  edm::LogInfo("HCalGeom") << "HcalDDDSimConstants: Maximum Layer for HB " 
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
      edm::LogInfo("HCalGeom") << "HcalDDDSimConstants: rHB[" << i << "] = "
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
      edm::LogInfo("HCalGeom") << "HcalDDDSimConstants: zHE[" << i << "] = "
			   << zHE[i] << " dzHE[" << i << "] = " << dzHE[i];
#endif
    }
  }

  nzHB   = (int)(izb.size());
  nmodHB = (int)(phib.size());
#ifdef DebugLog
  edm::LogInfo("HCalGeom") << "HcalDDDSimConstants::loadGeometry: " << nzHB
		       << " barrel half-sectors";
  for (int i=0; i<nzHB; i++)
    edm::LogInfo("HCalGeom") << "Section " << i << " Copy number " << izb[i];
  edm::LogInfo("HCalGeom") << "HcalDDDSimConstants::loadGeometry: " << nmodHB
		       << " barrel modules";
  for (int i=0; i<nmodHB; i++)
    edm::LogInfo("HCalGeom") << "Module " << i << " Copy number " << phib[i];
#endif

  nzHE   = (int)(ize.size());
  nmodHE = (int)(phie.size());
#ifdef DebugLog
  edm::LogInfo("HCalGeom") << "HcalDDDSimConstants::loadGeometry: " << nzHE
		       << " endcap half-sectors";
  for (int i=0; i<nzHE; i++)
    edm::LogInfo("HCalGeom") << "Section " << i << " Copy number " << ize[i];
  edm::LogInfo("HCalGeom") << "HcalDDDSimConstants::loadGeometry: " << nmodHE
		       << " endcap modules";
  for (int i=0; i<nmodHE; i++)
    edm::LogInfo("HCalGeom") << "Module " << i << " Copy number " << phie[i];
#endif

#ifdef DebugLog
  edm::LogInfo("HCalGeom") << "HO has Z of size " << zho.size();
  for (unsigned int kk=0; kk<zho.size(); kk++)
    edm::LogInfo("HCalGeom") << "ZHO[" << kk << "] = " << zho[kk];
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
  std::cout << "HO Eta boundaries " << etaHO[0] << " " << etaHO[1]
	    << " " << etaHO[2] << " " << etaHO[3] << std::endl;
  std::cout << "HO Parameters " << rminHO << " " << zho.size();
  for (int i=0; i<4; ++i) std::cout << " eta[" << i << "] = " << etaHO[i];
  for (unsigned int i=0; i<zho.size(); ++i) std::cout << " zho[" << i << "] = " << zho[i];
  std::cout << std::endl;
#endif
}

std::vector<double> HcalDDDSimConstants::getDDDArray(const std::string & str, 
						     const DDsvalues_type & sv,
						     int & nmin) const {
#ifdef DebugLog
  std::cout << "HcalDDDSimConstants:getDDDArray called for " << str
	    << " with nMin "  << nmin << std::endl;
#endif
  DDValue value(str);
  if (DDfetch(&sv,value)) {
#ifdef DebugLog
    std::cout << "HcalDDDSimConstants: " << value << std::endl;
#endif
    const std::vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nmin > 0) {
      if (nval < nmin) {
	edm::LogError("HCalGeom") << "HcalDDDSimConstants : # of " << str 
				  << " bins " << nval << " < " << nmin 
				  << " ==> illegal";
	throw cms::Exception("DDException") << "HcalDDDSimConstants: cannot get array " << str;
      }
    } else {
      if (nval < 1 && nmin == 0) {
	edm::LogError("HCalGeom") << "HcalDDDSimConstants : # of " << str
				  << " bins " << nval << " < 2 ==> illegal"
				  << " (nmin=" << nmin << ")";
	throw cms::Exception("DDException") << "HcalDDDSimConstants: cannot get array " << str;
      }
    }
    nmin = nval;
    return fvec;
  } else {
    if (nmin >= 0) {
      edm::LogError("HCalGeom") << "HcalDDDRecConstants: cannot get array "
				<< str;
      throw cms::Exception("DDException") << "HcalDDDRecConstants: cannot get array " << str;
    }
    std::vector<double> fvec;
    nmin = 0;
    return fvec;
  }
}

unsigned int HcalDDDSimConstants::find(int element, 
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

double HcalDDDSimConstants::deltaEta(int det, int etaR, int depth) const {

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
      } else if (det == static_cast<int>(HcalOuter)) {
	if (etaR == nOff[2]) {
	  tmp = 0.5*(etaHO[0]-etaTable[etaR-1]);
	} else if (etaR == nOff[2]+1) {
	  tmp = 0.5*(etaTable[etaR]-etaHO[1]);
	} else if (etaR == nOff[3]) {
	  tmp = 0.5*(etaHO[2]-etaTable[etaR-1]);
	} else if (etaR == nOff[3]+1) {
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
  edm::LogInfo("HCalGeom") << "HcalDDDSimConstants::deltaEta " << etaR << " " 
		       << depth << " ==> " << tmp;
#endif
  return tmp;
}

double HcalDDDSimConstants::getEta(int det, int etaR, int zside,
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
      } else if (det == static_cast<int>(HcalOuter)) {
	if (etaR == nOff[2]) {
	  tmp = 0.5*(etaHO[0]+etaTable[etaR-1]);
	} else if (etaR == nOff[2]+1) {
	  tmp = 0.5*(etaTable[etaR]+etaHO[1]);
	} else if (etaR == nOff[3]) {
	  tmp = 0.5*(etaHO[2]+etaTable[etaR-1]);
	} else if (etaR == nOff[3]+1) {
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
  edm::LogInfo("HCalGeom") << "HcalDDDSimConstants::getEta " << etaR << " " 
		       << zside << " " << depth << " ==> " << tmp;
#endif
  return tmp;
}
 
double HcalDDDSimConstants::getEta(double r, double z) const {

  double tmp = 0;
  if (z != 0) tmp = -log(tan(0.5*atan(r/z)));
#ifdef DebugLog
  edm::LogInfo("HCalGeom") << "HcalDDDSimConstants::getEta " << r << " " << z 
		       << " ==> " << tmp;
#endif
  return tmp;
}

int HcalDDDSimConstants::getShift(HcalSubdetector subdet, int depth) const {

  int shift;
  switch(subdet) {
  case HcalEndcap:
    shift = shiftHE[0];
    break;
  case HcalForward:
    shift = shiftHF[depth-1];
    break;
  case HcalOuter:
    shift = shiftHB[3];
    break;
  default:
    shift = shiftHB[0];
    break;
  }
  return shift;
}

double HcalDDDSimConstants::getGain(HcalSubdetector subdet, int depth) const {

  double gain;
  switch(subdet) {
  case HcalEndcap:
    gain = gainHE[0];
    break;
  case HcalForward:
    gain = gainHF[depth-1];
    break;
  case HcalOuter:
    gain = gainHB[3];
    break;
  default:
    gain = gainHB[0];
    break;
  }
  return gain;
}

void HcalDDDSimConstants::printTileHB(int eta, int depth) const {

  double etaL   = etaTable[eta-1];
  double thetaL = 2.*atan(exp(-etaL));
  double etaH   = etaTable[eta];
  double thetaH = 2.*atan(exp(-etaH));
  int    layL=0, layH=0;
  if (depth == 1) {
    layH = depths[0][eta-1];
  } else {
    layL = depths[0][eta-1];
    layH = depths[1][eta-1];
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

void HcalDDDSimConstants::printTileHE(int eta, int depth) const {

  double etaL   = etaTable[eta-1];
  double thetaL = 2.*atan(exp(-etaL));
  double etaH   = etaTable[eta];
  double thetaH = 2.*atan(exp(-etaH));
  int    layL=0, layH=0;
  if (eta == 16) {
    layH = depths[2][eta-1];
  } else if (depth == 1) {
    layH = depths[0][eta-1];
  } else if (depth == 2) {
    layL = depths[0][eta-1];
    layH = depths[1][eta-1];
  } else {
    layL = depths[1][eta-1];
    layH = depths[2][eta-1];
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
