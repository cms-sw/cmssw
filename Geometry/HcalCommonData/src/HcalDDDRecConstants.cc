#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#define DebugLog

HcalDDDRecConstants::HcalDDDRecConstants(const DDCompactView& cpv, 
					 const HcalDDDSimConstants& hcons) {

#ifdef DebugLog
  std::cout << "HcalDDDRecConstants::HcalDDDRecConstants (const DDCompactView& cpv, const HcalDDDSimConstants& hcons) constructor" << std::endl;
#endif

  std::string attribute = "OnlyForHcalRecNumbering"; 
  std::string value     = "any";
  DDValue val(attribute, value, 0.0);
  
  DDSpecificsFilter filter;
  filter.setCriteria(val, DDSpecificsFilter::not_equals,
		     DDSpecificsFilter::AND, true, // compare strings 
		     true  // use merged-specifics or simple-specifics
		     );
  DDFilteredView fv(cpv);
  fv.addFilter(filter);
  bool ok = fv.firstChild();

  if (ok) {
    //Load the SpecPars
    loadSpecPars(fv);

    //Load the Sim Constants
    loadSimConst(hcons);
  } else {
    edm::LogError("HCalGeom") << "HcalDDDRecConstants: cannot get filtered "
			      << " view for " << attribute << " not matching "
			      << value;
    throw cms::Exception("DDException") << "HcalDDDRecConstants: cannot match " << attribute << " to " << value;
  }
}


HcalDDDRecConstants::~HcalDDDRecConstants() { 
  //  std::cout << "destructed!!!" << std::endl;
}


HcalDDDRecConstants::HcalID HcalDDDRecConstants::getHCID(int subdet,int ieta,
							 int iphi, int lay,
							 int idepth) const {

  int    eta(ieta), phi(iphi), depth(idepth);
  if (subdet == static_cast<int>(HcalBarrel) || 
      subdet == static_cast<int>(HcalEndcap)) {
    eta  = ietaMap[ieta-1];
    phi  = (iphi-1)/phiGroup[eta-1]; ++phi;
    depth= layerGroup[eta-1][lay-1];
    if (eta == iEtaMin[1]) {
      if (subdet == static_cast<int>(HcalBarrel)) {
	if (depth > 2) depth = 2;
      } else {
	if (depth < 3) depth = 3;
      }
    } else if (eta == nOff[0] && lay > 1) {
      int   kphi   = phi + int((phioff[3]+0.1)/phibin[eta-1]);
      kphi         = (kphi-1)%4 + 1;
      if (kphi == 2 || kphi == 3) depth = layerGroup[eta-1][lay-2];
    } else if (eta == nOff[1] && depth > 2) {
       eta = nOff[1]-1;
    }
  } else if (subdet == static_cast<int>(HcalOuter)) {
    depth = 4;
  } 
  return HcalDDDRecConstants::HcalID(eta,phi,depth);
}

void HcalDDDRecConstants::loadSpecPars(const DDFilteredView& fv) {

  DDsvalues_type sv(fv.mergedSpecifics());
  
  char name[20];
  //Eta grouping
  nEta     = 0;
  sprintf (name, "etagroup");
  etaGroup = dbl_to_int(getDDDArray(name,sv,nEta));
#ifdef DebugLog
  std::cout << "HcalDDDRecConstants:Read etaGroup with " << nEta <<" members:";
  for (int i=0; i<nEta; i++) 
    std::cout << " [" << i << "] = " << etaGroup[i];
  std::cout << std::endl;
#endif

  //Phi Grouping
  sprintf (name, "phigroup");
  phiGroup = dbl_to_int(getDDDArray(name,sv,nEta));
#ifdef DebugLog
  std::cout << "HcalDDDRecConstants:Read phiGroup with " << nEta <<" members:";
  for (int i=0; i<nEta; i++) 
    std::cout << " [" << i << "] = " << phiGroup[i];
  std::cout << std::endl;
#endif

  //Layer grouping
  int layers = 19;
  for (int i=0; i<nEta; ++i) {
    sprintf (name, "layergroupEta%d", i+1);
    layerGroup[i] = dbl_to_int(getDDDArray(name,sv,layers));
    if (layers == 0) {
      layerGroup[i] = layerGroup[i-1]; 
      layers        = (int)(layerGroup[i].size());
    }
#ifdef DebugLog
    std::cout << "HcalDDDRecConstants:Read " << name << ":";
    for (int k=0; k<layers; k++) 
      std::cout << " [" << k << "] = " << layerGroup[i][k];
    std::cout << std::endl;
#endif
    layers = -1;
  }
}

void HcalDDDRecConstants::loadSimConst(const HcalDDDSimConstants& hcons) {

  for (int i=0; i<4; ++i) {
    std::pair<int,int> ieta = hcons.getiEtaRange(i);
    iEtaMin[i] = ieta.first;
    iEtaMax[i] = ieta.second;
    maxDepth[i]= 0;
  }
  maxDepth[2] = 4;
  maxDepth[3] = 4;

  // First eta table
  std::vector<double> etas = hcons.getEtaTable();
  etaTable.clear(); ietaMap.clear();
  int ieta(0), ietaHB(0), ietaHE(0);
  etaTable.push_back(etas[ieta]);
  for (int i=0; i<nEta; ++i) {
    ieta += etaGroup[i];
    if (ieta >= (int)(etas.size())) {
      edm::LogError("HCalGeom") << "Going beyond the array boundary "
				<< etas.size() << " at index " << i 
				<< " of etaTable from SimConstant";
      throw cms::Exception("DDException") << "Going beyond the array boundary "
					  << etas.size() << " at index " << i 
					  << " of etaTable from SimConstant";
    } else {
      etaTable.push_back(etas[ieta]);
    }
    for (int k=0; k<etaGroup[i]; ++k) ietaMap.push_back(i+1);
    if (ieta <= iEtaMax[0]) ietaHB = i+1;
    if (ieta <= iEtaMin[1]) ietaHE = i+1;
    iEtaMax[1] = i+1;
  }
  iEtaMin[1] = ietaHE;
  iEtaMax[0] = ietaHB;

  // Then Phi bins
  ieta = 0;
  phibin.clear(); 
  for (int i=0; i<nEta; ++i) {
    double dphi = phiGroup[i]*hcons.getPhiBin(ieta);
    phibin.push_back(dphi);
    ieta += etaGroup[i];
  }
#ifdef DebugLog
  std::cout << "Modified eta/deltaphi table for " << nEta << " bins" << std::endl;
  for (int i=0; i<nEta; ++i) 
    std::cout << "Eta[" << i << "] = " << etaTable[i] << ":" << etaTable[i+1]
	      << " PhiBin[" << i << "] = " << phibin[i]/CLHEP::deg <<std::endl;
#endif

  //Phi offsets for barrel and endcap & special constants
  phioff.clear();
  for (int i=0; i<4; ++i)
    phioff.push_back(hcons.getPhiOff(i));
  nOff = hcons.getNOff();

  //Now the depths
  for (int i=0; i<nEta; ++i) {
    unsigned int imx = layerGroup[i].size();
    int laymax = (imx > 0) ? layerGroup[i][imx-1] : 0;
    if (i < iEtaMax[0]) {
      int laymax0 = (imx > 16) ? layerGroup[i][16] : laymax;
      std::cout << "HB " << i << " " << imx << " " << laymax << " " << laymax0 << std::endl;
      if (maxDepth[0] < laymax0) maxDepth[0] = laymax0;
    }
    if (i >= iEtaMin[1]-1 && i < iEtaMax[1]) {
      std::cout << "HE " << i << " " << imx << " " << laymax << std::endl;
      if (maxDepth[1] < laymax) maxDepth[1] = laymax;
    }
  }
#ifdef DebugLog
  for (int i=0; i<4; ++i) 
    std::cout << "Detector Type[" << i << "] iEta " << iEtaMin[i] << ":"
	      << iEtaMax[i] << " MaxDepth " << maxDepth[i] << std::endl; 
#endif
}

std::vector<double> HcalDDDRecConstants::getDDDArray(const char * str, 
						     const DDsvalues_type & sv,
						     int & nmin) const {
#ifdef DebugLog
  std::cout << "HcalDDDRecConstants:getDDDArray called for " << str
	    << " with nMin "  << nmin << std::endl;
#endif
  DDValue value(str);
  if (DDfetch(&sv,value)) {
#ifdef DebugLog
    std::cout << "HcalDDDRecConstants: " << value << std::endl;
#endif
    const std::vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nmin > 0) {
      if (nval < nmin) {
	edm::LogError("HCalGeom") << "HcalDDDRecConstants : # of " << str 
				  << " bins " << nval << " < " << nmin 
				  << " ==> illegal";
	throw cms::Exception("DDException") << "HcalDDDRecConstants: cannot get array " << str;
      }
    } else {
      if (nval < 1 && nmin == 0) {
	edm::LogError("HCalGeom") << "HcalDDDRecConstants : # of " << str
				  << " bins " << nval << " < 1 ==> illegal"
				  << " (nmin=" << nmin << ")";
	throw cms::Exception("DDException") << "HcalDDDRecConstants: cannot get array " << str;
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
