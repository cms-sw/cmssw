#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

//#define DebugLog

unsigned int
HcalDDDSimConstants::layerGroupSize( unsigned int eta ) const
{
  unsigned int k = 0;
  for( auto const & it : hpar->layerGroupEtaSim )
  {
    if( it.layer == eta + 1 )
    {
      return it.layerGroup.size();
    }
    if( it.layer > eta + 1 )
      break;
    k = it.layerGroup.size();
  }
  return k;
}

unsigned int
HcalDDDSimConstants::layerGroup( unsigned int eta, unsigned int i ) const
{
  unsigned int k = 0;
  for( auto const & it :  hpar->layerGroupEtaSim )
  {
    if( it.layer == eta + 1 )
    {
      return it.layerGroup.at( i );
    }
    if( it.layer > eta + 1 )
      break;

    k = it.layerGroup.at( i );
  }
  return k;
}

HcalDDDSimConstants::HcalDDDSimConstants(const HcalParameters* hp) : hpar(hp) {

#ifdef DebugLog
  edm::LogInfo("HCalGeom") << "HcalDDDSimConstants::HcalDDDSimConstants (const HcalParameter* hp) constructor";
#endif

  initialize();
#ifdef DebugLog
  std::vector<HcalCellType> cellTypes = HcalCellTypes();
  edm::LogInfo ("HCalGeom") << "HcalDDDSimConstants: " << cellTypes.size()
			    << " cells of type HCal (All)";
#endif
}


HcalDDDSimConstants::~HcalDDDSimConstants() { 
#ifdef DebugLog
  edm::LogInfo ("HCalGeom") << "HcalDDDSimConstants::destructed!!!";
#endif
}

HcalCellType::HcalCell HcalDDDSimConstants::cell(int idet, int zside, 
						 int depth, int etaR,
						 int iphi) const {

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
  else if (idet == static_cast<int>(HcalBarrel) && depth > 17)              ok=false;
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
    phi  = fioff + (iphi - 0.5)*fibin;
    dphi = 0.5*fibin;
    if (idet == static_cast<int>(HcalForward)) {
      int ir = nR + hpar->etaMin[2] - etaR - 1;
      if (ir > 0 && ir < nR) {
	rz     = 0.5*(hpar->rTable[ir]+hpar->rTable[ir-1]);
	drz    = 0.5*(hpar->rTable[ir]-hpar->rTable[ir-1]);
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


std::pair<int,double> HcalDDDSimConstants::getDetEta(double eta, int depth) {

  int    hsubdet(0), ieta(0);
  double etaR(0);
  double heta = fabs(eta);
  for (int i = 0; i < nEta; i++)
    if (heta > hpar->etaTable[i]) ieta = i + 1;
  if (heta <= hpar->etaRange[1]) {
    if ((ieta <= hpar->etaMin[1] && depth==3) || ieta > hpar->etaMax[0]) {
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
      if (lay == 18) {
	if (hetaR > etaHO[1] && ieta == hpar->noff[2]) ieta++;
      }
    } else {
      if (ieta <= hpar->etaMin[1]) ieta = hpar->etaMin[1];
    }
  }
  return ieta;
}

std::pair<int,int> HcalDDDSimConstants::getEtaDepth(int det, int etaR, int phi,
						    int depth, int lay) {

  //Modify the depth index
  if (det == static_cast<int>(HcalForward)) { // Forward HCal
  } else if (det == static_cast<int>(HcalOuter)) {
    depth = 4;
  } else {
    if (lay >= 0) {
      depth= layerGroup( etaR-1, lay-1 );
      if (etaR == hpar->noff[0] && lay > 1) {
	int   kphi   = phi + int((hpar->phioff[3]+0.1)/hpar->phibin[etaR-1]);
	kphi         = (kphi-1)%4 + 1;
	if (kphi == 2 || kphi == 3) depth = layerGroup( etaR-1, lay-2 );
      }
    } else if (det == static_cast<int>(HcalBarrel)) {
      if (depth==3) depth = 2;
    }
    if (etaR == hpar->noff[1] && depth > 2) {
      etaR = hpar->noff[1]-1;
    } else if (etaR == hpar->etaMin[1]) {
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
#ifdef DebugLog
    edm::LogInfo ("HCalGeom") << "R " << r << " Z " << z << " eta " << etaR 
			      << ":" << eta;
    if (eta != etaR) edm::LogInfo ("HCalGeom") << "**** Check *****";
#endif
    return eta;
  } else {
    return etaR;
  }
}

unsigned int HcalDDDSimConstants::findLayer(int layer, const std::vector<HcalParameters::LayerItem>& layerGroup) const {
  
  unsigned int id = layerGroup.size();
  for (unsigned int i = 0; i < layerGroup.size(); i++) {
    if (layer == (int)(layerGroup[i].layer)) {
      id = i;
      break;
    }
  }
  return id;
}

std::pair<int,int> HcalDDDSimConstants::getModHalfHBHE(const int type) const {

  if (type == 0) {
    return std::pair<int,int>(nmodHB,nzHB);
  } else {
    return std::pair<int,int>(nmodHE,nzHE);
  }
}

std::pair<double,double> HcalDDDSimConstants::getPhiCons(int det, int ieta) {

  double fioff(0), fibin(0);
  if (det == static_cast<int>(HcalForward)) { // Forward HCal
    fioff   = hpar->phioff[2];
    fibin   = hpar->phitable[ieta-hpar->etaMin[2]];
    if  (unitPhi(fibin) > 2) {   // HF double-phi  
      fioff = hpar->phioff[4];
    }
  } else { // Barrel or Endcap
    if (det == static_cast<int>(HcalBarrel)) {
      fioff   = hpar->phioff[0];
    } else {
      fioff   = hpar->phioff[1];
    }
    fibin = hpar->phibin[ieta-1];
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
  int ietamin = (ieta>0) ? ieta : hpar->etaMin[indx];
  int ietamax = (ieta>0) ? ieta : hpar->etaMax[indx];

  int phi = 1, zside  = 1;

  // Get the Cells 
  int subdet0 = static_cast<int>(subdet);
  for (int depth=dmin; depth<=dmax; depth++) {
    int    shift = getShift(subdet, depth);
    double gain  = getGain (subdet, depth);
    if (subdet == HcalForward) {
      if (depth%2 == 1) hsize = dzVcal;
      else              hsize = dzVcal-0.5*dlShort;
    }
    for (int eta=ietamin; eta<= ietamax; eta++) {
      HcalCellType::HcalCell temp1 = cell(subdet0,zside,depth,eta,phi);
      if (temp1.ok) {
	int units = unitPhi (subdet0, eta);
	HcalCellType temp2(subdet, eta, phi, depth, temp1,
			   shift, gain, nz, nmod, hsize, units);
	if (subdet == HcalOuter) {
	  if (eta == hpar->noff[4]) {
	    std::vector<int> missPlus, missMinus;
	    int kk = 7;
	    for (int miss=0; miss<hpar->noff[5]; miss++) {
	      missPlus.push_back(hpar->noff[kk]);
	      kk++;
	    }
	    for (int miss=0; miss<hpar->noff[6]; miss++) {
	      missMinus.push_back(hpar->noff[kk]);
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
 
  std::cout << "Tile Information for HB from " << hpar->etaMin.at(0) << " to " << hpar->etaMax.at(0) << "\n\n";
  for (int eta=hpar->etaMin.at(0); eta<= hpar->etaMax.at(0); eta++) {
    int dmax = 1;
    if (depths[0][eta-1] < 17) dmax = 2;
    for (int depth=1; depth<=dmax; depth++) 
      printTileHB(eta, depth);
  }

  std::cout << "\nTile Information for HE from " << hpar->etaMin.at(1) << " to " << hpar->etaMax.at(1) << "\n\n";
  for (int eta=hpar->etaMin[1]; eta<= hpar->etaMax[1]; eta++) {
    int dmin=1, dmax=3;
    if (eta == hpar->etaMin[1]) {
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

  double dphi = (det == static_cast<int>(HcalForward)) ? hpar->phitable[etaR-hpar->etaMin[2]] : hpar->phibin[etaR-1];
  return unitPhi(dphi);
}

int HcalDDDSimConstants::unitPhi(double dphi) const {

  const double fiveDegInRad = 2*M_PI/72;
  int units = int(dphi/fiveDegInRad+0.5);
  return units;
}

void HcalDDDSimConstants::initialize( void ) {

  nEta      = hpar->etaTable.size();
  nR        = hpar->rTable.size();
  nPhiF     = nR - 1;

#ifdef DebugLog
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
#ifdef DebugLog
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
      int laymax0 = (imx > 16) ? layerGroup( i, 16 ) : laymax;
      if (i+1 == hpar->etaMax[0] && laymax0 > 2) laymax0 = 2;
      if (maxDepth[0] < laymax0) maxDepth[0] = laymax0;
    }
    if (i >= hpar->etaMin[1]-1 && i < hpar->etaMax[1]) {
      if (maxDepth[1] < laymax) maxDepth[1] = laymax;
    }
  }
#ifdef DebugLog
  for (int i=0; i<4; ++i)
    std::cout << "Detector Type [" << i << "] iEta " << hpar->etaMin[i] << ":" 
	      << hpar->etaMax[i] << " MaxDepth " << maxDepth[i] << std::endl;
#endif

  int maxdepth = (maxDepth[1]>maxDepth[0]) ? maxDepth[1] : maxDepth[0];
  for (int i=0; i<maxdepth; ++i) {
    for (int k=0; k<nEta-1; ++k) {
      int layermx = ((k+1 < hpar->etaMin[1]) && i < maxDepth[0]) ? 17 : (int)layerGroupSize( k );
      int ll      = layermx;
      for (int l=layermx-1; l >= 0; --l) {
	if ((int)layerGroup( k, l ) == i+1) {
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

  nzHB   = hpar->modHB[0];
  nmodHB = hpar->modHB[1];
  nzHE   = hpar->modHE[0];
  nmodHE = hpar->modHE[1];
#ifdef DebugLog
  std::cout << "HcalDDDSimConstants:: " << nzHB << ":" << nmodHB
	    << " barrel and " << nzHE << ":" << nmodHE
	    << " endcap half-sectors" << std::endl;
#endif

  if (hpar->rHB.size() > 17 && hpar->zHO.size() > 4) {
    rminHO   = hpar->rHO[0];
    for (int k=0; k<4; ++k) etaHO[k] = hpar->rHO[k+1];
  } else {
    rminHO   =-1.0;
    etaHO[0] = hpar->etaTable[4];
    etaHO[1] = hpar->etaTable[4];
    etaHO[2] = hpar->etaTable[10];
    etaHO[3] = hpar->etaTable[10];
  }
#ifdef DebugLog
  std::cout << "HO Eta boundaries " << etaHO[0] << " " << etaHO[1]
	    << " " << etaHO[2] << " " << etaHO[3] << std::endl;
  std::cout << "HO Parameters " << rminHO << " " << hpar->zHO.size();
  for (int i=0; i<4; ++i) std::cout << " eta[" << i << "] = " << etaHO[i];
  for (unsigned int i=0; i<hpar->zHO.size(); ++i) 
    std::cout << " zHO[" << i << "] = " << hpar->zHO[i];
  std::cout << std::endl;
#endif
}

double HcalDDDSimConstants::deltaEta(int det, int etaR, int depth) const {

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
      if (etaR == hpar->noff[1]-1 && depth > 2) {
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
    int ir = nR + hpar->etaMin[2] - etaR - 1;
    if (ir > 0 && ir < nR) {
      double z = zVcal;
      if (depth%2 != 1) z += dlShort;
      tmp = 0.5*(getEta(hpar->rTable[ir-1],z)+getEta(hpar->rTable[ir],z));
    }
  } else {
    if (etaR > 0 && etaR < nEta) {
      if (etaR == hpar->noff[1]-1 && depth > 2) {
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
    shift = hpar->HEShift[0];
    break;
  case HcalForward:
    shift = hpar->HFShift[depth-1];
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

double HcalDDDSimConstants::getGain(HcalSubdetector subdet, int depth) const {

  double gain;
  switch(subdet) {
  case HcalEndcap:
    gain = hpar->HEGains[0];
    break;
  case HcalForward:
    gain = hpar->HFGains[depth-1];
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

void HcalDDDSimConstants::printTileHB(int eta, int depth) const {
  std::cout << "HcalDDDSimConstants::printTileHB for eta " << eta << " and depth " << depth << "\n";
  
  double etaL   = hpar->etaTable.at(eta-1);
  double thetaL = 2.*atan(exp(-etaL));
  double etaH   = hpar->etaTable.at(eta);
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

void HcalDDDSimConstants::printTileHE(int eta, int depth) const {

  double etaL   = hpar->etaTable[eta-1];
  double thetaL = 2.*atan(exp(-etaL));
  double etaH   = hpar->etaTable[eta];
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
  double phib  = hpar->phibin[eta-1];
  int nphi = 2;
  if (phib > 6*CLHEP::deg) nphi = 1;
  std::cout << "\ntileHE:: Eta/depth " << eta << "|" << depth << " theta " << thetaH/CLHEP::deg << ":" << thetaL/CLHEP::deg << " Layer " << layL << ":" << layH-1 << " phi " << nphi << "\n";
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
