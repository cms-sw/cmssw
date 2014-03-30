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

//#define DebugLog

HcalDDDRecConstants::HcalDDDRecConstants() : tobeInitialized(true), hcons(0) {

#ifdef DebugLog
  std::cout << "HcalDDDRecConstants::HcalDDDRecConstants() constructor" << std::endl;
#endif
}

HcalDDDRecConstants::HcalDDDRecConstants(const DDCompactView& cpv, const HcalDDDSimConstants& hconst) : tobeInitialized(true), hcons(0) {

#ifdef DebugLog
  std::cout << "HcalDDDRecConstants::HcalDDDRecConstants (const DDCompactView& cpv, const HcalDDDSimConstants& hconst) constructor" << std::endl;
#endif
  initialize(cpv, hconst);
}

HcalDDDRecConstants::~HcalDDDRecConstants() { 
  //  std::cout << "destructed!!!" << std::endl;
}

std::vector<HcalDDDRecConstants::HcalEtaBin> 
HcalDDDRecConstants::getEtaBins(const int itype) const {

  checkInitialized();
  std::vector<HcalDDDRecConstants::HcalEtaBin> bins;
  unsigned int type  = (itype == 0) ? 0 : 1;
  unsigned int lymax = (type == 0) ? 17 : 19;
  for (int ieta = iEtaMin[type]; ieta <= iEtaMax[type]; ++ieta) {
    int nfi = (int)((20.001*nModule[itype]*CLHEP::deg)/phibin[ieta-1]);
    HcalDDDRecConstants::HcalEtaBin etabin = HcalDDDRecConstants::HcalEtaBin(ieta,etaTable[ieta-1], etaTable[ieta], nfi, phioff[type], phibin[ieta-1]);
    int dstart = -1;
    if (layerGroup[ieta-1].size() > 0) {
      int lmin(0), lmax(0);
      int dep = layerGroup[ieta-1][0];
      if (type == 1 && ieta == iEtaMin[type]) dep = 3;
      unsigned lymx0 = (layerGroup[ieta-1].size() > lymax) ? lymax : layerGroup[ieta-1].size();
      for (unsigned int l=0; l<lymx0; ++l) {
	if (layerGroup[ieta-1][l] == dep) {
	  if (lmin == 0) lmin = l + 1;
	  lmax = l + 1;
	} else if (layerGroup[ieta-1][l] > dep) {
	  if (dstart < 0) dstart = dep;
	  etabin.layer.push_back(std::pair<int,int>(lmin,lmax));
	  lmin = (l + 1);
	  lmax = l;
	  dep  = layerGroup[ieta-1][l];
	}
	if (type == 0 && ieta == iEtaMax[type] && dep > 2) break;
      }
      if (lmax >= lmin) {
	if (ieta+1 == nOff[1]) {
	} else if (ieta == nOff[1]) {
	  HcalDDDRecConstants::HcalEtaBin etabin0 = HcalDDDRecConstants::HcalEtaBin(ieta-1,etaTable[ieta-2], etaTable[ieta], nfi, phioff[type], phibin[ieta-1]);
	  etabin0.depthStart = dep;
	  etabin0.layer.push_back(std::pair<int,int>(lmin,lmax));
	  bins.push_back(etabin0);
	} else {
	  etabin.layer.push_back(std::pair<int,int>(lmin,lmax));
	  if (dstart < 0) dstart = dep;
	}
      }
    }
    etabin.depthStart = dstart;
    bins.push_back(etabin);
  }
#ifdef DebugLog
  std::cout << "Prepares " << bins.size() << " eta bins for type " << type 
	    << std::endl;
  for (unsigned int i=0; i<bins.size(); ++i) {
    std::cout << "Bin[" << i << "]: Eta = (" << bins[i].ieta << ":"
	      << bins[i].etaMin << ":" << bins[i].etaMax << ") Phi = (" 
	      << bins[i].nPhi << ":" << bins[i].phi0 << ":" << bins[i].dphi 
	      << ") and " << bins[i].layer.size() << " depths (start) "
	      << bins[i].depthStart << " :";
    for (unsigned int k=0; k<bins[i].layer.size(); ++k)
      std::cout << " [" << k << "] " << bins[i].layer[k].first << ":"
		<< bins[i].layer[k].second;
    std::cout << std::endl;
  }
#endif
  return bins;
}

HcalDDDRecConstants::HcalID HcalDDDRecConstants::getHCID(int subdet,int ieta,
							 int iphi, int lay,
							 int idepth) const {

  checkInitialized();
  int    eta(ieta), phi(iphi), depth(idepth);
  if ((subdet == static_cast<int>(HcalOuter)) ||
      ((subdet == static_cast<int>(HcalBarrel)) && (lay > 17))) {
    subdet= static_cast<int>(HcalOuter);
    depth = 4;
  } else if (subdet == static_cast<int>(HcalBarrel) || 
      subdet == static_cast<int>(HcalEndcap)) {
    eta      = ietaMap[ieta-1];
    int unit = phiUnitS[ieta-1];
    int phi0 = (iphi-1)/phiGroup[eta-1];
    if (unit == 2) {
      phi0   = (iphi+1)/2;
      phi0   = (phi0-1)/phiGroup[eta-1];
    } else if (unit == 4) {
      phi0   = (iphi+5)/4;
      phi0   = (phi0-1)/phiGroup[eta-1];
    }
    ++phi0;
    unit     = hcons->unitPhi(phibin[eta-1]);
    phi      = hcons->phiNumber(phi0,unit);
    depth    = layerGroup[eta-1][lay-1];
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
  } 
#ifdef DebugLog
  std::cout << "getHCID: input " << subdet << ":" << ieta << ":" << iphi
	    << ":" << idepth << ":" << lay << " output " << eta << ":" << phi
	    << ":" << depth << std::endl;
#endif
  return HcalDDDRecConstants::HcalID(subdet,eta,phi,depth);
}

std::vector<HcalCellType> HcalDDDRecConstants::HcalCellTypes(HcalSubdetector subdet) const {

  checkInitialized();
  if (subdet == HcalBarrel || subdet == HcalEndcap) {
    std::vector<HcalCellType> cells;
    int isub   = (subdet == HcalBarrel) ? 0 : 1;
    std::vector<HcalDDDRecConstants::HcalEtaBin> etabins = getEtaBins(isub);
    for (unsigned int bin=0; bin<etabins.size(); ++bin) {
      std::vector<HcalCellType> temp;
      std::vector<int>          count;
      std::vector<double>       dmin, dmax;
      for (unsigned int il=0; il<etabins[bin].layer.size(); ++il) {
	HcalCellType cell(subdet, 0, 0, 0, HcalCellType::HcalCell());
	temp.push_back(cell);
	count.push_back(0);
	dmin.push_back(0);
	dmax.push_back(0);
      }
      int ieta = etabins[bin].ieta;
      for (int keta=etaSimValu[ieta-1].first; keta<=etaSimValu[ieta-1].second;
	   ++keta) {
	std::vector<HcalCellType> cells = hcons->HcalCellTypes(subdet,keta,-1);
	for (unsigned int ic=0; ic<cells.size(); ++ic) {
	  for (unsigned int il=0; il<etabins[bin].layer.size(); ++il) {
	    if (cells[ic].depthSegment() >= etabins[bin].layer[il].first &&
		cells[ic].depthSegment() <= etabins[bin].layer[il].second) {
	      if (count[il] == 0) {
		temp[il] = cells[ic];
		dmin[il] = cells[ic].depthMin();
		dmax[il] = cells[ic].depthMax();
	      }
	      ++count[il];
	      if (cells[ic].depthMin() < dmin[il]) dmin[il] = cells[ic].depthMin();
	      if (cells[ic].depthMax() > dmax[il]) dmax[il] = cells[ic].depthMax();
	      break;
	    }
	  }
	}
      }
      int unit = hcons->unitPhi(etabins[bin].dphi);
      for (unsigned int il=0; il<etabins[bin].layer.size(); ++il) {
	int depth = etabins[bin].depthStart + (int)(il);
	temp[il].setEta(ieta,etabins[bin].etaMin,etabins[bin].etaMax);
	temp[il].setPhi(etabins[bin].nPhi,unit,etabins[bin].dphi/CLHEP::deg,
			phioff[isub]/CLHEP::deg);
	temp[il].setDepth(depth,dmin[il],dmax[il]);
	cells.push_back(temp[il]);
      }
    }
#ifdef DebugLog
    std::cout << "HcalDDDRecConstants: found " << cells.size() << " cells for sub-detector type " << isub << std::endl;
    for (unsigned int ic=0; ic<cells.size(); ++ic)
      std::cout << "Cell[" << ic << "] " << cells[ic] << std::endl;
#endif
    return cells;
  } else {
    return hcons->HcalCellTypes(subdet,-1,-1);
  }
}

void HcalDDDRecConstants::initialize(const DDCompactView& cpv, 
				     const HcalDDDSimConstants& hconst) {
  if (tobeInitialized) {
    tobeInitialized = false;
    hcons           = &(hconst);
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
      loadSimConst();
    } else {
      edm::LogError("HCalGeom") << "HcalDDDRecConstants: cannot get filtered "
				<< " view for " << attribute << " not matching "
				<< value;
      throw cms::Exception("DDException") << "HcalDDDRecConstants: cannot match " << attribute << " to " << value;
    }
  }
}

unsigned int HcalDDDRecConstants::numberOfCells(HcalSubdetector subdet) const {

  checkInitialized();
  if (subdet == HcalBarrel || subdet == HcalEndcap) {
    unsigned int num = 0;
    std::vector<HcalCellType> cellTypes = HcalCellTypes(subdet);
    for (unsigned int i=0; i<cellTypes.size(); i++) {
      num += (unsigned int)(cellTypes[i].nPhiBins());
      if (cellTypes[i].nHalves() > 1) 
	num += (unsigned int)(cellTypes[i].nPhiBins());
      num -= (unsigned int)(cellTypes[i].nPhiMissingBins());
    }
#ifdef DebugLog
    edm::LogInfo ("HCalGeom") << "HcalDDDRecConstants:numberOfCells " 
			      << cellTypes.size()  << " " << num 
			      << " for subdetector " << subdet;
#endif
    return num;
  } else {
    return hcons->numberOfCells(subdet);
  }

}

unsigned int HcalDDDRecConstants::nCells(HcalSubdetector subdet) const {

  checkInitialized();
  if (subdet == HcalBarrel || subdet == HcalEndcap) {
    int isub   = (subdet == HcalBarrel) ? 0 : 1;
    std::vector<HcalDDDRecConstants::HcalEtaBin> etabins = getEtaBins(isub);
    unsigned int ncell(0);
    for (unsigned int i=0; i<etabins.size(); ++i) {
      ncell += (((unsigned int)(etabins[i].nPhi))*(etabins[i].layer.size()));
    }
    return ncell;
  } else if (subdet == HcalOuter) {
    return kHOSizePreLS1;
  } else if (subdet == HcalForward) {
    return kHFSizePreLS1;
  } else {
    return 0;
  }
}

unsigned int HcalDDDRecConstants::nCells() const {
  checkInitialized();
  return (nCells(HcalBarrel)+nCells(HcalEndcap)+nCells(HcalOuter)+nCells(HcalForward));
}

void HcalDDDRecConstants::checkInitialized() const {
  if (tobeInitialized) {
    edm::LogError("HcalGeom") << "HcalDDDRecConstants : to be initialized correctly";
    throw cms::Exception("DDException") << "HcalDDDRecConstants: to be initialized";
  }
} 

void HcalDDDRecConstants::loadSpecPars(const DDFilteredView& fv) {

  DDsvalues_type sv(fv.mergedSpecifics());
  
  //Topology Mode
  modeTopo = getDDDString("TopologyMode",sv);

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

void HcalDDDRecConstants::loadSimConst() {

  for (int i=0; i<4; ++i) {
    std::pair<int,int> ieta = hcons->getiEtaRange(i);
    iEtaMin[i] = ieta.first;
    iEtaMax[i] = ieta.second;
    maxDepth[i]= 0;
  }
  maxDepth[2] = 4;
  maxDepth[3] = 4;

  // First eta table
  std::vector<double> etas = hcons->getEtaTable();
  etaTable.clear(); ietaMap.clear(); etaSimValu.clear();
  int ieta(0), ietaHB(0), ietaHE(0);
  etaTable.push_back(etas[ieta]);
  for (int i=0; i<nEta; ++i) {
    int ef = ieta+1;
    ieta  += etaGroup[i];
    if (ieta >= (int)(etas.size())) {
      edm::LogError("HCalGeom") << "Going beyond the array boundary "
				<< etas.size() << " at index " << i 
				<< " of etaTable from SimConstant";
      throw cms::Exception("DDException") << "Going beyond the array boundary "
					  << etas.size() << " at index " << i 
					  << " of etaTable from SimConstant";
    } else {
      etaTable.push_back(etas[ieta]);
      etaSimValu.push_back(std::pair<int,int>(ef,ieta));
    }
    for (int k=0; k<etaGroup[i]; ++k) ietaMap.push_back(i+1);
    if (ieta <= iEtaMax[0]) ietaHB = i+1;
    if (ieta <= iEtaMin[1]) ietaHE = i+1;
    iEtaMax[1] = i+1;
  }
  iEtaMin[1] = ietaHE;
  iEtaMax[0] = ietaHB;
  etaTableHF = hcons->getEtaTableHF();

  // Then Phi bins
  ieta = 0;
  phibin.clear(); phiUnitS.clear();
  for (int i=0; i<nEta; ++i) {
    double dphi = phiGroup[i]*hcons->getPhiBin(ieta);
    phibin.push_back(dphi);
    ieta += etaGroup[i];
  }
  for (unsigned int i=1; i<etas.size(); ++i) {
    int unit = hcons->unitPhi(hcons->getPhiBin(i-1));
    phiUnitS.push_back(unit);
  }
  phibinHF = hcons->getPhiTableHF();
#ifdef DebugLog
  std::cout << "Modified eta/deltaphi table for " << nEta << " bins" << std::endl;
  for (int i=0; i<nEta; ++i) 
    std::cout << "Eta[" << i << "] = " << etaTable[i] << ":" << etaTable[i+1]
	      << ":" << etaSimValu[i].first << ":" << etaSimValu[i].second
	      << " PhiBin[" << i << "] = " << phibin[i]/CLHEP::deg <<std::endl;
  std::cout << "PhiUnitS";
  for (unsigned int i=0; i<phiUnitS.size(); ++i)
    std::cout << " [" << i << "] = " << phiUnitS[i];
  std::cout << std::endl;
  std::cout << "EtaTableHF";
  for (unsigned int i=0; i<etaTableHF.size(); ++i)
    std::cout << " [" << i << "] = " << etaTableHF[i];
  std::cout << std::endl;
  std::cout << "PhiBinHF";
  for (unsigned int i=0; i<phibinHF.size(); ++i)
    std::cout << " [" << i << "] = " << phibinHF[i];
  std::cout << std::endl;
#endif

  //Phi offsets for barrel and endcap & special constants
  phioff = hcons->getPhiOffs();
  nOff   = hcons->getNOff();

  //Now the depths
  for (int i=0; i<nEta; ++i) {
    unsigned int imx = layerGroup[i].size();
    int laymax = (imx > 0) ? layerGroup[i][imx-1] : 0;
    if (i < iEtaMax[0]) {
      int laymax0 = (imx > 16) ? layerGroup[i][16] : laymax;
      if (i+1 == iEtaMax[0] && laymax0 > 2) laymax0 = 2;
#ifdef DebugLog
      std::cout << "HB " << i << " " << imx << " " << laymax << " " << laymax0 << std::endl;
#endif
      if (maxDepth[0] < laymax0) maxDepth[0] = laymax0;
    }
    if (i >= iEtaMin[1]-1 && i < iEtaMax[1]) {
#ifdef DebugLog
      std::cout << "HE " << i << " " << imx << " " << laymax << std::endl;
#endif
      if (maxDepth[1] < laymax) maxDepth[1] = laymax;
    }
  }
#ifdef DebugLog
  for (int i=0; i<4; ++i) 
    std::cout << "Detector Type[" << i << "] iEta " << iEtaMin[i] << ":"
	      << iEtaMax[i] << " MaxDepth " << maxDepth[i] << std::endl; 
#endif

  //Now the geometry constants
  std::pair<int,int> nmodz = hcons->getModHalfHBHE(0);
  nModule[0] = nmodz.first;
  nHalves[0] = nmodz.second;
  gconsHB    = hcons->getConstHBHE(0);
  for (unsigned int i=0; i<gconsHB.size(); ++i) {
    gconsHB[i].first  /= CLHEP::cm;
    gconsHB[i].second /= CLHEP::cm;
  }
#ifdef DebugLog
  std::cout << "HB with " << nModule[0] << " modules and " << nHalves[0]
	    <<" halves and " << gconsHB.size() << " layers" << std::endl;
  for (unsigned int i=0; i<gconsHB.size(); ++i) 
    std::cout << "rHB[" << i << "] = " << gconsHB[i].first << " +- "
	      << gconsHB[i].second << std::endl; 
#endif
  nmodz      = hcons->getModHalfHBHE(1);
  nModule[1] = nmodz.first;
  nHalves[1] = nmodz.second;
  gconsHE= hcons->getConstHBHE(1);
  for (unsigned int i=0; i<gconsHE.size(); ++i) {
    gconsHE[i].first  /= CLHEP::cm;
    gconsHE[i].second /= CLHEP::cm;
  }
#ifdef DebugLog
  std::cout << "HE with " << nModule[1] << " modules and " << nHalves[1] 
	    <<" halves and " << gconsHE.size() << " layers" << std::endl;
  for (unsigned int i=0; i<gconsHE.size(); ++i) 
    std::cout << "zHE[" << i << "] = " << gconsHE[i].first << " +- "
	      << gconsHE[i].second << std::endl; 
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

std::string HcalDDDRecConstants::getDDDString(const std::string & str, 
					      const DDsvalues_type & sv) const{
#ifdef DebugLog
  std::cout << "HcalDDDRecConstants:getDDDString called for "<<str <<std::endl;
#endif
  DDValue value(str);
  if (DDfetch(&sv,value)) {
#ifdef DebugLog
    std::cout << "HcalDDDRecConstants: " << value << std::endl;
#endif
    const std::vector<std::string> & fvec = value.strings();
    if (fvec.size() == 0) {
      edm::LogError("HCalGeom") << "HcalDDDRecConstants : # of " << str 
				<< " bins is 0 ==> illegal";
      throw cms::Exception("DDException") << "HcalDDDRecConstants: cannot get array " << str;
    }
    return fvec[0];
  } else {
    edm::LogError("HCalGeom") << "HcalDDDRecConstants: cannot get array "<<str;
    throw cms::Exception("DDException") << "HcalDDDRecConstants: cannot get array " << str;
  }
}
