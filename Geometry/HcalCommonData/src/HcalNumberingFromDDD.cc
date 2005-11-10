///////////////////////////////////////////////////////////////////////////////
// File: HcalNumberingFromDDD.cc
// Description: Usage of DDD to get to numbering scheme for hadron calorimeter
///////////////////////////////////////////////////////////////////////////////

namespace std{} using namespace std;
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "Geometry/HcalCommonData/interface/HcalNumberingFromDDD.h"

#include "DetectorDescription/Base/interface/DDException.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDValue.h"

#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"
#include <iostream>

#define debug

HcalNumberingFromDDD::HcalNumberingFromDDD(std::string & name,
					   const DDCompactView & cpv,
					   edm::ParameterSet const & p) {
  edm::ParameterSet m_HCalDD = p.getParameter<edm::ParameterSet>("HcalNumberingFromDDD");
  verbosity= m_HCalDD.getParameter<int>("Verbosity");
  if (verbosity>0) std::cout << "Creating HcalNumberingFromDDD" << std::endl;
  initialize(name, cpv);
}

HcalNumberingFromDDD::~HcalNumberingFromDDD() {
  if (verbosity>0) std::cout << "Deleting HcalNumberingFromDDD" << std::endl;
}

HcalNumberingFromDDD::HcalID HcalNumberingFromDDD::unitID(int det,
							  Hep3Vector point,
							  int depth,
							  int lay) const {

  double hx  = point.x();
  double hy  = point.y();
  double hz  = point.z();
  double hR  = sqrt(hx*hx+hy*hy+hz*hz);
  double htheta = (hR == 0. ? 0. : acos(max(min(hz/hR,1.0),-1.0)));
  double hsintheta = sin(htheta);
  double hphi = (hR*hsintheta == 0. ? 0. :atan2(hy,hx));
  double heta = (abs(hsintheta) == 1.? 0. : -log(abs(tan(htheta/2.))) );
  heta = abs(heta);

  int etaR, hsubdet=0;
  double fibin;

  //First eta index
  if (det == 5) { // Forward HCal
    hsubdet = static_cast<int>(HcalForward);
    hR      = sqrt(hx*hx+hy*hy);
    hphi   += phioff[2];
    etaR    = etaMax[2];
    for (int i = nR-1; i > 0; i--)
      if (hR < rTable[i]) etaR = etaMin[2] + nR - i - 1;
    fibin   = phibin[nEta+etaR-etaMin[2]-1];
  } else { // Barrel or Endcap
    etaR  = 1;
    for (int i = 0; i < nEta-1; i++)
      if (heta > etaTable[i]) etaR = i + 1;
    if (det == 3) {
      hsubdet = static_cast<int>(HcalBarrel);
      hphi   += phioff[0];
      if (etaR > etaMax[0])  etaR = etaMax[0];
    } else {
      hsubdet = static_cast<int>(HcalEndcap);
      hphi   += phioff[1];
      if (etaR <= etaMin[1]) etaR = etaMin[1];
    }
    fibin = phibin[etaR-1];
  }
  if (hphi < 0)     hphi += twopi;

#ifdef debug
  if (verbosity>1)
    std::cout << "HcalNumberingFromDDD: point = " << point << " eta " << heta
	      << " phi " << hphi << std::endl;
#endif
  //Then the phi index
  int nphi  = int((twopi+0.1*fibin)/fibin);
  int zside = hz>0 ? 1: 0;
  int iphi  = int(hphi/fibin) + 1;
  if (iphi > nphi) iphi = 1;

  HcalNumberingFromDDD::HcalID tmp = unitID(hsubdet,zside,depth,etaR,iphi,lay);
  return tmp;
}

HcalNumberingFromDDD::HcalID HcalNumberingFromDDD::unitID(double eta,double fi,
							  int depth, int lay) const {

  int    etaR = 0;
  double heta = abs(eta);
  for (int i = 0; i < nEta; i++)
    if (heta > etaTable[i]) etaR = i + 1;
  int    hsubdet=0;
  double fibin, fioff;
  if (etaR <= etaMin[1]) {
    if ((etaR <= etaMin[1] && depth==3) || etaR > etaMax[0]) {
      hsubdet = static_cast<int>(HcalEndcap);
      fioff   = phioff[1];
    } else {
      hsubdet = static_cast<int>(HcalBarrel);
      fioff   = phioff[0];
    }
    fibin = phibin[etaR-1];
  } else {
    hsubdet = static_cast<int>(HcalForward);
    fioff   = phioff[2];
    double theta = 2.*atan(exp(-heta));
    double hR    = zVcal*tan(theta);
    etaR    = etaMax[2];
    for (int i = nR-1; i > 0; i--)
      if (hR < rTable[i]) etaR = etaMin[2] + nR - i - 1;
    fibin   = phibin[nEta+etaR-etaMin[2]-1];
  }

  int    nphi  = int((twopi+0.1*fibin)/fibin);
  int    zside = eta>0 ? 1: 0;
  double hphi  = fi+fioff;
  if (hphi < 0)    hphi += twopi;
  int    iphi  = int(hphi/fibin) + 1;
  if (iphi > nphi) iphi = 1;

  HcalNumberingFromDDD::HcalID tmp = unitID(hsubdet,zside,depth,etaR,iphi,lay);
  return tmp;
}

  
HcalNumberingFromDDD::HcalID HcalNumberingFromDDD::unitID(int det, int zside,
							  int depth, int etaR,
							  int phi, int lay) const {

  //Mpdify the depth index
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
  if (etaR == nOff[1] && depth > 2) etaR = nOff[1]-1;

  HcalNumberingFromDDD::HcalID tmp(det,zside,depth,etaR,phi,lay);
#ifdef debug
  if (verbosity>1)
    std::cout << "HcalNumberingFromDDD: det = " << det << " " << tmp.subdet 
	      << " zside = " << tmp.zside << " depth = " << tmp.depth 
	      << " eta/R = " << tmp.etaR << " phi = " << tmp.phi
	      << " layer = " << tmp.lay << std::endl;
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
       idet==static_cast<int>(HcalForward)) && etaR >=etaMn && etaR <= etaMx)
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
    if      (idet == static_cast<int>(HcalBarrel)) {
      fioff = phioff[0];
      fibin = phibin[etaR-1];
    } else if (idet == static_cast<int>(HcalEndcap)) {
      fioff = phioff[1];
      fibin = phibin[etaR-1];
    } else {
      fioff = phioff[2];
      fibin = phibin[nEta+etaR-etaMin[2]-1];
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
#ifdef debug
	if (verbosity>1)
	  std::cout << "HcalNumberingFromDDD: wrong eta " << etaR << " (" 
		    << ir << "/" << nR << ") Detector " << idet << std::endl;
#endif
      }
      if (depth != 1 && depth != 2) {
	ok     = false;
#ifdef debug
	if (verbosity>1)
	  std::cout << "HcalNumberingFromDDD: wrong depth " << depth 
		    << " in Detector " << idet << std::endl;
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
	laymax = depth0;
      } else if (depth == 2) {
	laymin = depth0+1;
        laymax = depth2[etaR-1];
      } else  if (depth == 3) {
	laymin = depth2[etaR-1]+1;
        laymax = depth3[etaR-1];
	if (etaR<=etaMin[1] && idet==static_cast<int>(HcalEndcap)) laymin=1;
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
#ifdef debug
	if (verbosity>1)
	  std::cout << "HcalNumberingFromDDD: wrong depth " << depth 
		    << " (Layer minimum " << laymin << " maximum "
		    << laymax << " maxLay " << maxlay << ")" << std::endl;
#endif
      }
    } else {
      ok = false;
#ifdef debug
      if (verbosity>1)
	std::cout << "HcalNumberingFromDDD: wrong eta " << etaR
		  << "/" << nEta << " Detector " << idet << std::endl;
#endif
    }
  } else {
    ok = false;
#ifdef debug
    if (verbosity>1)
      std::cout << "HcalNumberingFromDDD: wrong eta " << etaR << " det "
		<< idet << std::endl;
#endif
  }
  HcalCellType::HcalCell tmp(ok,eta,deta,phi,dphi,rz,drz,flagrz);

#ifdef debug
  if (verbosity>2)
    std::cout << "HcalNumberingFromDDD: det/side/depth/etaR/phi " << det 
	      << "/" << zside << "/" << depth << "/" << etaR << "/" << iphi
	      << " Cell Flag " << tmp.ok << " " << tmp.eta << " " << tmp.deta
	      << " phi " << tmp.phi << " " << tmp.dphi << " r(z) " << tmp.rz 
	      << " " << tmp.drz << " " << tmp.flagrz << std::endl;
#endif
  return tmp;
}

vector<double> HcalNumberingFromDDD::getEtaTable() const {

  vector<double> tmp = etaTable;
  return tmp;
}

vector<HcalCellType::HcalCellType> HcalNumberingFromDDD::HcalCellTypes() const{

  vector<HcalCellType::HcalCellType> cellTypes;
  int phi = 1, zside  = 1, subdet;
  bool cor = false;

  // Get the Cells for HB 
  subdet = static_cast<int>(HcalBarrel);
  for (int depth=1; depth<=4; depth++) {
    int    shift = shiftHB[depth-1];
    double gain  = gainHB[depth-1];
    for (int eta=etaMin[0]; eta<= etaMax[0]; eta++) {
      HcalCellType::HcalCell temp1 = cell(subdet, zside, depth, eta, phi, cor);
      if (temp1.ok) {
	HcalCellType::HcalCellType temp2(subdet, eta, phi, depth, temp1,
					 shift, gain);
	cellTypes.push_back(temp2);
      }
    }
  }

  // Get the Cells for HE
  subdet = static_cast<int>(HcalEndcap);
  for (int depth=1; depth<=3; depth++) {
    int    shift = shiftHE[depth-1];
    double gain  = gainHE[depth-1];
    for (int eta=etaMin[1]; eta<= etaMax[1]; eta++) {
      HcalCellType::HcalCell temp1 = cell(subdet, zside, depth, eta, phi, cor);
      if (temp1.ok) {
	HcalCellType::HcalCellType temp2(subdet, eta, phi, depth, temp1,
					 shift, gain);
	cellTypes.push_back(temp2);
      }
    }
  }

  // Get the Cells for HF
  subdet = static_cast<int>(HcalForward);
  for (int depth=1; depth<=2; depth++) {
    int    shift = shiftHF[depth-1];
    double gain  = gainHF[depth-1];
    for (int eta=etaMin[2]; eta<= etaMax[2]; eta++) {
      HcalCellType::HcalCell temp1 = cell(subdet, zside, depth, eta, phi, cor);
      if (temp1.ok) {
	HcalCellType::HcalCellType temp2(subdet, eta, phi, depth, temp1,
					 shift, gain);
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
#ifdef debug
  if (verbosity>2)
    std::cout << "HcalNumberingFromDDD::getEta " << etaR << " " << zside 
	      << " " << depth << " ==> " << tmp << std::endl;
#endif
  return tmp;
}
 
double HcalNumberingFromDDD::getEta(double r, double z) const {

  double tmp = 0;
  if (z != 0) tmp = -log(tan(0.5*atan(r/z)));
#ifdef debug
  if (verbosity>2)
    std::cout << "HcalNumberingFromDDD::getEta " << r << " " << z 
	      << " ==> " << tmp << std::endl;
#endif
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
#ifdef debug
  if (verbosity>2)
    std::cout << "HcalNumberingFromDDD::deltaEta " << etaR << " " << depth 
	      << " ==> " << tmp << std::endl;
#endif
  return tmp;
}

void HcalNumberingFromDDD::initialize(std::string & name, 
				      const DDCompactView & cpv) {

  std::string attribute = "ReadOutName";
  if (verbosity>0) 
    std::cout << "HcalNumberingFromDDD: Initailise for " << name << " as "
	      << attribute << std::endl;

  DDSpecificsFilter filter;
  DDValue           ddv(attribute,name,0);
  filter.setCriteria(ddv,DDSpecificsFilter::equals);
  DDFilteredView fv(cpv);
  fv.addFilter(filter);
  fv.firstChild();

  //Load the SpecPars
  loadSpecPars(fv);

  //Load the Geometry parameters
  loadGeometry(fv);
}

void HcalNumberingFromDDD::loadSpecPars(DDFilteredView fv) {
  DDsvalues_type sv(fv.mergedSpecifics());

  // Phi Offset
  int i, nphi=4;
  vector<double> tmp1 = getDDDArray("phioff",sv,nphi);
#ifdef debug
  if (verbosity>1)
    std::cout << "HcalNumberingFromDDD: " << nphi << " phioff";
#endif
  phioff.resize(tmp1.size());
  for (i=0; i<nphi; i++) {
    phioff[i] = tmp1[i];
#ifdef debug
    if (verbosity>1) std::cout << " " << phioff[i]/deg;
#endif
  }
#ifdef debug
  if (verbosity>1) std::cout << std::endl;
#endif

  //Eta table
  nEta     = -1;
  vector<double> tmp2 = getDDDArray("etaTable",sv,nEta);
#ifdef debug
  if (verbosity>1)
    std::cout << "HcalNumberingFromDDD: " << nEta << " etaTable" << std::endl 
	      << "                    ";
#endif
  etaTable.resize(tmp2.size());
  for (i=0; i<nEta; i++) {
    etaTable[i] = tmp2[i];
#ifdef debug
    if (verbosity>1) {
      std::cout << " " << i << " " << etaTable[i];
      if (i%10 == 9) std::cout << std::endl << "                    ";
    }
#endif
  }
#ifdef debug
  if (verbosity>1) std::cout << std::endl;
#endif

  //R table
  nR     = -1;
  vector<double> tmp3 = getDDDArray("rTable",sv,nR);
#ifdef debug
  if (verbosity>1) 
    std::cout << "HcalNumberingFromDDD: " << nR << " rTable (cm)";
#endif
  rTable.resize(tmp3.size());
  for (i=0; i<nR; i++) {
    rTable[i] = tmp3[i];
#ifdef debug
    if (verbosity>1) std::cout << " " << i << rTable[i]/cm;
#endif
  }
#ifdef debug
  if (verbosity>1) std::cout << std::endl;
#endif

  //Phi bins
  nPhi   = nEta + nR - 2;
  vector<double> tmp4 = getDDDArray("phibin",sv,nPhi);
#ifdef debug
  if (verbosity>1) 
    std::cout << "HcalNumberingFromDDD: " << nPhi << " phiBin";
#endif
  phibin.resize(tmp4.size());
  for (i=0; i<nPhi; i++) {
    phibin[i] = tmp4[i];
#ifdef debug
    if (verbosity>1) {
      std::cout << " " << i << " " << phibin[i]/deg;
      if (i%10 == 9) std::cout << std::endl << "                      ";
    }
#endif
  }
#ifdef debug
  if (verbosity>1) std::cout << std::endl;
#endif

  //Layer boundaries for depths 1, 2, 3, 4
  nDepth            = nEta - 1;
  vector<double> d1 = getDDDArray("depth1",sv,nDepth);
  nDepth            = nEta - 1;
  vector<double> d2 = getDDDArray("depth2",sv,nDepth);
  nDepth            = nEta - 1;
  vector<double> d3 = getDDDArray("depth3",sv,nDepth);
#ifdef debug
  if (verbosity>1) std::cout << "HcalNumberingFromDDD: " << nDepth << " Depth";
#endif
  depth1.resize(nDepth);
  depth2.resize(nDepth);
  depth3.resize(nDepth);
  for (i=0; i<nDepth; i++) {
    depth1[i] = static_cast<int>(d1[i]);
    depth2[i] = static_cast<int>(d2[i]);
    depth3[i] = static_cast<int>(d3[i]);
#ifdef debug
    if (verbosity>1) {
      std::cout << " " << i << " " << depth1[i] << " " << depth2[i] << " "
		<< depth3[i];
      if (i%8 == 7) std::cout << std::endl << "                      ";
    }
#endif
  }
#ifdef debug
  if (verbosity>1) std::cout << std::endl;
#endif

  // Minimum and maximum eta boundaries
  int             ndx  = 3;
  vector<double>  tmp5 = getDDDArray("etaMin",sv,ndx);
  vector<double>  tmp6 = getDDDArray("etaMax",sv,ndx);
  etaMin.resize(ndx);
  etaMax.resize(ndx);
  for (i=0; i<ndx; i++) {
    etaMin[i] = static_cast<int>(tmp5[i]);
    etaMax[i] = static_cast<int>(tmp6[i]);
  }
  etaMin[0] = 1;
  etaMax[1] = nEta-1;
  etaMax[2] = etaMin[2]+nR-2;
#ifdef debug
  if (verbosity>1) {
    std::cout << "HcalNumberingFromDDD: etaMin/etaMax " << ndx << " values";
    for (i=0; i<ndx; i++) {
      std::cout << " # " << i << " min " << etaMin[i] << " max " << etaMax[i];
    }
    std::cout << std::endl;
  }
#endif

  // Geometry parameters for HF
  int ngpar = 7;
  vector<double> gpar = getDDDArray("gparHF",sv,ngpar);
  zVcal = gpar[6];
#ifdef debug
  if (verbosity>1) 
    std::cout << "HcalNumberingFromDDD: zVcal " << zVcal << std::endl;
#endif

  // nOff
  int             noff = 3;
  vector<double>  nvec = getDDDArray("noff",sv,noff);
#ifdef debug
  if (verbosity>1) 
    std::cout << "HcalNumberingFromDDD: Noff " << noff << " values";
#endif
  nOff.resize(noff);
  for (i=0; i<noff; i++) {
    nOff[i] = static_cast<int>(nvec[i]);
#ifdef debug
    if (verbosity>1) std::cout << " # " << i << " nOff " << nOff[i];
#endif
  }
#ifdef debug
  if (verbosity>1) std::cout << std::endl;
#endif

  //Gains and Shifts for HB depths
  ndx                  = 4;
  gainHB               = getDDDArray("HBGains",sv,ndx);
  vector<double>  tmp7 = getDDDArray("HBShift",sv,ndx);
  shiftHB.resize(ndx);
#ifdef debug
  if (verbosity>1) 
    std::cout << "HcalNumberingFromDDD:: Gain factor and Shift for HB depth "
	      << "layers:";
#endif
  for (i=0; i<ndx; i++) {
    shiftHB[i] = static_cast<int>(tmp7[i]);
#ifdef debug
    if (verbosity>1) 
      std::cout << " " << i << " " << gainHB[i] << " " << shiftHB[i];
#endif
  }
#ifdef debug
  if (verbosity>1) std::cout << std::endl;
#endif

  //Gains and Shifts for HB depths
  ndx                  = 4;
  gainHE               = getDDDArray("HEGains",sv,ndx);
  vector<double>  tmp8 = getDDDArray("HEShift",sv,ndx);
  shiftHE.resize(ndx);
#ifdef debug
  if (verbosity>1) 
    std::cout << "HcalNumberingFromDDD:: Gain factor and Shift for HE depth "
	      << "layers:";
#endif
  for (i=0; i<ndx; i++) {
    shiftHE[i] = static_cast<int>(tmp8[i]);
#ifdef debug
    if (verbosity>1) 
      std::cout << " " << i << " " << gainHE[i] << " " << shiftHE[i];
#endif
  }
#ifdef debug
  if (verbosity>1) std::cout << std::endl;
#endif

  //Gains and Shifts for HF depths
  ndx                  = 4;
  gainHF               = getDDDArray("HFGains",sv,ndx);
  vector<double>  tmp9 = getDDDArray("HFShift",sv,ndx);
  shiftHF.resize(ndx);
#ifdef debug
  if (verbosity>1)
    std::cout << "HcalNumberingFromDDD:: Gain factor and Shift for HF depth "
	      << "layers:";
#endif
  for (i=0; i<ndx; i++) {
    shiftHF[i] = static_cast<int>(tmp9[i]);
#ifdef debug
    if (verbosity>1) 
      std::cout << " " << i << " " << gainHF[i] << " " << shiftHF[i];
#endif
  }
#ifdef debug
  if (verbosity>1) std::cout << std::endl;
#endif
}

void HcalNumberingFromDDD::loadGeometry(DDFilteredView fv) {
  
  bool dodet=true, hf=false;
  vector<double> rb(20,0.0), ze(20,0.0);
  vector<int>    ib(20,0),   ie(20,0);
  double         zf = 0;
  dzVcal = -1.;           

  while (dodet) {
    DDTranslation   t    = fv.translation();
    vector<int>     copy = fv.copyNumbers();
    const DDSolid & sol  = fv.logicalPart().solid();
    int idet = 0, lay = -1;
    int nsiz = (int)(copy.size());
    if (nsiz>0) lay  = copy[nsiz-1]/10;
    if (nsiz>1) idet = copy[nsiz-2]/1000;
    if (idet == 3) {
      // HB
#ifdef debug
      if (verbosity>2) 
	std::cout << "HB " << sol.name() << " Shape " << sol.shape()
		  << " Layer " << lay << " R " << t.perp() << std::endl;
#endif
      if (lay >=0 && lay < 20) {
	ib[lay]++;
	rb[lay] += t.perp();
      }
    } else if (idet == 4) {
      // HE
#ifdef debug
      if (verbosity>2) 
	std::cout << "HE " << sol.name() << " Shape " << sol.shape()
		  << " Layer " << lay << " Z " << t.z() << std::endl;
#endif
      if (lay >=0 && lay < 20) {
	ie[lay]++;
	ze[lay] += abs(t.z());
      }
    } else if (idet == 5) {
      // HF
      if (!hf) {
	const std::vector<double> & paras = sol.parameters();
#ifdef debug
	if (verbosity>2) {
	  std::cout << "HF " << sol.name() << " Shape " << sol.shape()
		    << " Z " << t.z() << " Parameters";
	  for (unsigned j=0; j<paras.size(); j++)
	    std::cout << " " << paras[j];
	  std::cout << std::endl;
	}
#endif
	zf  = abs(t.z());
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
#ifdef debug
      if (verbosity>2) 
	std::cout << "Unknown Detector " << idet << " for " << sol.name() 
		  << " Shape " << sol.shape() << " R " << t.perp() << " Z " 
		  << t.z() << std::endl;
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
  }

#ifdef debug
  if (verbosity>1) 
    std::cout << "HcalNumberingFromDDD: Maximum Layer for HB " << ibmx 
	      << " for HE " << iemx << " Z for HF " << zf << " extent "
	      << dzVcal << std::endl << "                      R for HB";
#endif
  if (ibmx > 0) {
    rHB.resize(ibmx);
    for (int i=0; i<ibmx; i++) {
      rHB[i] = rb[i];
#ifdef debug
      if (verbosity>1) std::cout << " " << rHB[i];
#endif
    }
  }
#ifdef debug
  if (verbosity>1) std::cout << std::endl << "                      Z for HE";
#endif
  if (iemx > 0) {
    zHE.resize(iemx);
    for (int i=0; i<iemx; i++) {
      zHE[i] = ze[i];
#ifdef debug
      if (verbosity>1) std::cout << " " << zHE[i];
#endif
    }
  }
#ifdef debug
  if (verbosity>1) std::cout << std::endl;
#endif
}

vector<double> HcalNumberingFromDDD::getDDDArray(const string & str, 
						 const DDsvalues_type & sv, 
						 int & nmin) const {
#ifdef debug
  if (verbosity>1) 
    std::cout << "HcalNumberingFromDDD:getDDDArray called for " << str 
	      << " with nMin "  << nmin << std::endl;
#endif
  DDValue value(str);
  if (DDfetch(&sv,value)) {
#ifdef debug
    if (verbosity>2) std::cout << value << " " << std::endl;
#endif
    const vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nmin > 0) {
      if (nval < nmin) {
	if (verbosity>0) 
	  std::cout << "HcalNumberingFromDDD : # of " << str << " bins " 
		    << nval << " < " << nmin << " ==> illegal" << std::endl;
	throw DDException("HcalNumberingFromDDD: cannot get array "+str);
      }
    } else {
      if (nval < 2) {
	if (verbosity>0) 
	  std::cout << "HcalNumberingFromDDD : # of " << str << " bins " 
		    << nval << " < 2 ==> illegal (nmin=" << nmin << ")" 
		    << std::endl;
	throw DDException("HcalNumberingFromDDD: cannot get array "+str);
      }
    }
    nmin = nval;
    return fvec;
  } else {
    throw DDException("HcalNumberingFromDDD: cannot get array "+str);
  }
}
