///////////////////////////////////////////////////////////////////////////////
// File: HcalNumberingFromDDD.cc
// Description: Usage of DDD to get to numbering scheme for hadron calorimeter
///////////////////////////////////////////////////////////////////////////////

#include "CondFormats/GeometryObjects/interface/HcalParameters.h"
#include "Geometry/HcalCommonData/interface/HcalNumberingFromDDD.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <iostream>

//#define DebugLog

HcalNumberingFromDDD::HcalNumberingFromDDD(HcalDDDSimConstants *hcons) :
  hcalConstants(hcons) {
  edm::LogInfo("HCalGeom") << "Creating HcalNumberingFromDDD";
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
      etaR    = hcalConstants->getEtaHO(heta,hx,hy,hz);
    } else {
      hsubdet = static_cast<int>(HcalEndcap);
    }
  }

#ifdef DebugLog
  LogDebug("HCalGeom") << "HcalNumberingFromDDD: point = " << point << " det "
		       << det << ":" << hsubdet << " eta/R " << etaR << " phi "
		       << hphi;
#endif
  return unitID(hsubdet,etaR,hphi,depth,lay);
}

HcalNumberingFromDDD::HcalID HcalNumberingFromDDD::unitID(double eta,double fi,
							  int depth, 
							  int lay) const {
  std::pair<int,double> detEta = hcalConstants->getDetEta(eta, depth);
  return unitID(detEta.first,detEta.second,fi,depth,lay);
}


HcalNumberingFromDDD::HcalID HcalNumberingFromDDD::unitID(int det,
							  double etaR,
							  double phi,
							  int depth,
							  int lay) const {

  double hetaR = fabs(etaR);
  int    ieta  = hcalConstants->getEta(det, lay, hetaR);
  std::pair<double,double> ficons = hcalConstants->getPhiCons(det, ieta);

  int    nphi  = int((CLHEP::twopi+0.1*ficons.second)/ficons.second);
  int    zside = etaR>0 ? 1: 0;
  double hphi  = phi+ficons.first;
  if (hphi < 0)    hphi += CLHEP::twopi;
  int    iphi  = int(hphi/ficons.second) + 1;
  if (iphi > nphi) iphi = 1;

#ifdef DebugLog
  LogDebug("HCalGeom") << "HcalNumberingFromDDD: etaR = " << etaR << " : "
                       << zside << "/" << ieta << " phi " << hphi << " : "
                       << iphi;
#endif
  return unitID(det,zside,depth,ieta,iphi,lay);
}

HcalNumberingFromDDD::HcalID HcalNumberingFromDDD::unitID(int det, int zside,
							  int depth, int etaR,
							  int phi, 
							  int lay) const {


  std::pair<int,int> etaDepth = hcalConstants->getEtaDepth(det, etaR, phi, depth, lay);
  if (det == static_cast<int>(HcalBarrel) && etaDepth.second == 4) {
    det = static_cast<int>(HcalOuter);
  }

  int units     = hcalConstants->unitPhi(det, etaDepth.first);
  int iphi_skip = hcalConstants->phiNumber(phi, units);

#ifdef DebugLog
  LogDebug("HCalGeom") << "HcalNumberingFromDDD: phi units= " << units  
                       << "  iphi_skip= " << iphi_skip; 
#endif
  HcalNumberingFromDDD::HcalID tmp(det,zside,etaDepth.second,etaDepth.first,phi,iphi_skip,lay);

#ifdef DebugLog
  LogDebug("HCalGeom") << "HcalNumberingFromDDD: det = " << det << " " 
                       << tmp.subdet << " zside = " << tmp.zside << " depth = "
                       << tmp.depth << " eta/R = " << tmp.etaR << " phi = " 
                       << tmp.phi << " layer = " << tmp.lay;
#endif
  return tmp;
}
