#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

//#define DebugLog
enum { kHOSizePreLS1 = 2160, kHFSizePreLS1 = 1728 } ;

HcalDDDRecConstants::HcalDDDRecConstants(const HcalParameters* hp,
					 const HcalDDDSimConstants& hc) : 
  hpar(hp), hcons(hc) {

#ifdef DebugLog
  std::cout << "HcalDDDRecConstants::HcalDDDRecConstants (const HcalParameters* hp) constructor" << std::endl;
#endif
  initialize();
}

HcalDDDRecConstants::~HcalDDDRecConstants() { 
#ifdef DebugLog
  std::cout << "HcalDDDRecConstants::destructed!!!" << std::endl;
#endif
}

std::vector<HcalDDDRecConstants::HcalEtaBin> 
HcalDDDRecConstants::getEtaBins(const int itype) const {

  std::vector<HcalDDDRecConstants::HcalEtaBin> bins;
  unsigned int type  = (itype == 0) ? 0 : 1;
  unsigned int lymax = (type == 0) ? 17 : 19;
  for (int ieta = iEtaMin[type]; ieta <= iEtaMax[type]; ++ieta) {
    int nfi = (int)((20.001*nModule[itype]*CLHEP::deg)/phibin[ieta-1]);
    HcalDDDRecConstants::HcalEtaBin etabin = HcalDDDRecConstants::HcalEtaBin(ieta, etaTable[ieta-1], etaTable[ieta], nfi, hpar->phioff[type], phibin[ieta-1]);
    int dstart = -1;
    if (layerGroupSize( ieta-1 ) > 0) {
      int lmin(0), lmax(0);
      int dep = layerGroup( ieta-1, 0 );
      if (type == 1 && ieta == iEtaMin[type]) dep = 3;
      unsigned lymx0 = (layerGroupSize( ieta-1 ) > lymax) ? lymax : layerGroupSize( ieta-1 );
      for (unsigned int l=0; l<lymx0; ++l) {
	if ((int)layerGroup( ieta-1, l ) == dep) {
	  if (lmin == 0) lmin = l + 1;
	  lmax = l + 1;
	} else if ((int)layerGroup( ieta-1, l ) > dep) {
	  if (dstart < 0) dstart = dep;
	  etabin.layer.push_back(std::pair<int,int>(lmin,lmax));
	  lmin = (l + 1);
	  lmax = l;
	  dep  = layerGroup( ieta-1, l );
	}
	if (type == 0 && ieta == iEtaMax[type] && dep > 2) break;
      }
      if (lmax >= lmin) {
	if (ieta+1 == hpar->noff[1]) {
	} else if (ieta == hpar->noff[1]) {
	  HcalDDDRecConstants::HcalEtaBin etabin0 = HcalDDDRecConstants::HcalEtaBin(ieta-1, etaTable[ieta-2], etaTable[ieta], nfi, hpar->phioff[type], phibin[ieta-1]);
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

std::pair<double,double> 
HcalDDDRecConstants::getEtaPhi(int subdet, int ieta, int iphi) const {
  int ietaAbs = (ieta > 0) ? ieta : -ieta;
  const double fiveDegInRad = 2*M_PI/72;
  double eta(0), phi(0);
  if ((subdet == static_cast<int>(HcalBarrel)) || 
      (subdet == static_cast<int>(HcalEndcap)) ||
      (subdet == static_cast<int>(HcalOuter))) {  // Use Eta Table
    int unit    = (int)(phibin[ietaAbs-1]/fiveDegInRad+0.5);
    int kphi    = (unit == 2) ? ((iphi-1)/2 + 1) : iphi;
    double foff = (ietaAbs <= iEtaMax[0]) ? hpar->phioff[0] : hpar->phioff[1];
    eta         = 0.5*(etaTable[ietaAbs-1]+etaTable[ietaAbs]);
    phi         = foff + (kphi-0.5)*phibin[ietaAbs-1];
  } else {
    ietaAbs    -= iEtaMin[3];
    int unit    = (int)(hpar->phitable[ietaAbs-1]/fiveDegInRad+0.5);
    int kphi    = (unit == 4) ? ((iphi-3)/4 + 1) : ((iphi-1)/2 + 1);
    double foff = (unit > 2) ? hpar->phioff[4] : hpar->phioff[2];
    eta         = 0.5*(hpar->etaTableHF[ietaAbs-1]+hpar->etaTableHF[ietaAbs]);
    phi         = foff + (kphi-0.5)*hpar->phitable[ietaAbs-1];
  }
  if (ieta < 0)   eta  = -eta;
  if (phi > M_PI) phi -= (2*M_PI);
#ifdef DebugLog
  std::cout << "getEtaPhi: subdet|ieta|iphi " << subdet << "|" << ieta << "|"
	    << iphi << " eta|phi " << eta << "|" << phi << std::endl;
#endif
  return std::pair<double,double>(eta,phi);
}

HcalDDDRecConstants::HcalID 
HcalDDDRecConstants::getHCID(int subdet, int ieta, int iphi, int lay,
			     int idepth) const {

  int    eta(ieta), phi(iphi), depth(idepth);
  if ((subdet == static_cast<int>(HcalOuter)) ||
      ((subdet == static_cast<int>(HcalBarrel)) && (lay > 17))) {
    subdet= static_cast<int>(HcalOuter);
    depth = 4;
  } else if (subdet == static_cast<int>(HcalBarrel) || 
      subdet == static_cast<int>(HcalEndcap)) {
    eta      = ietaMap[ieta-1];
    int unit = phiUnitS[ieta-1];
    int phi0 = (iphi-1)/(hpar->phigroup[eta-1]);
    if (unit == 2) {
      phi0   = (iphi+1)/2;
      phi0   = (phi0-1)/(hpar->phigroup[eta-1]);
    } else if (unit == 4) {
      phi0   = (iphi+5)/4;
      phi0   = (phi0-1)/(hpar->phigroup[eta-1]);
    }
    ++phi0;
    unit     = hcons.unitPhi(phibin[eta-1]);
    phi      = hcons.phiNumber(phi0,unit);
    depth    = layerGroup( eta-1, lay-1 );
    if (eta == iEtaMin[1]) {
      if (subdet == static_cast<int>(HcalBarrel)) {
	if (depth > 2) depth = 2;
      } else {
	if (depth < 3) depth = 3;
      }
    } else if (eta == hpar->noff[0] && lay > 1) {
      int   kphi   = phi + int((hpar->phioff[3]+0.1)/phibin[eta-1]);
      kphi         = (kphi-1)%4 + 1;
      if (kphi == 2 || kphi == 3) depth = layerGroup( eta-1, lay-2 );
    } else if (eta == hpar->noff[1] && depth > 2) {
       eta = hpar->noff[1]-1;
    }
  } 
#ifdef DebugLog
  std::cout << "getHCID: input " << subdet << ":" << ieta << ":" << iphi
	    << ":" << idepth << ":" << lay << " output " << eta << ":" << phi
	    << ":" << depth << std::endl;
#endif
  return HcalDDDRecConstants::HcalID(subdet,eta,phi,depth);
}


double HcalDDDRecConstants::getRZ(int subdet, int ieta, int depth) const {

  int ietaAbs = (ieta > 0) ? ieta : -ieta;
  double rz(0);
#ifdef DebugLog
  int    lay(0);
#endif
  if (ietaAbs < hpar->etaMax[1]) {
    for (unsigned int k=0; k< layerGroupSize( ietaAbs-1 ); ++k) {
      if (depth == (int)layerGroup( ietaAbs-1, k )) {
	rz = ((subdet == static_cast<int>(HcalBarrel)) ? (gconsHB[k].first) :
	      (gconsHE[k].first));
	if (rz > 10.) {
#ifdef DebugLog
	  lay = k;
#endif
	  break;
	}
      }
    }
  }
#ifdef DebugLog
  std::cout << "getRZ: subdet|ieta|depth " << subdet << "|" << ieta << "|"
	    << depth << " lay|rz " << lay << "|" << rz << std::endl;
#endif
  return rz;
}
 	
std::vector<HcalDDDRecConstants::HcalActiveLength> 
HcalDDDRecConstants::getThickActive(const int type) const {

  std::vector<HcalDDDRecConstants::HcalActiveLength> actives;
  std::vector<HcalDDDRecConstants::HcalEtaBin> bins = getEtaBins(type);
#ifdef DebugLog
  unsigned int kount(0);
#endif
  for (unsigned int k=0; k<bins.size(); ++k) {
    int    ieta  = bins[k].ieta;
    double eta   = 0.5*(bins[k].etaMin+bins[k].etaMax);
    double theta = 2*atan(exp(-eta));
    double scale = 1.0/((type == 0) ? sin(theta) : cos(theta));
    int    depth = bins[k].depthStart;
    for (unsigned int i = 0; i < bins[k].layer.size(); ++i) {
      double thick(0);
      for (int j = bins[k].layer[i].first; j <= bins[k].layer[i].second; ++j) {
	if (type == 0 || j > 1) 
	  thick += ((type == 0) ? gconsHB[j-1].second : gconsHE[j-1].second);
      }
      thick *= (2.*scale);
      HcalDDDRecConstants::HcalActiveLength active(ieta,depth,eta,thick);
      actives.push_back(active);
      ++depth;
#ifdef DebugLog
      kount++;
      std::cout << "getThickActive: [" << kount << "] eta:" << active.ieta 
		<< ":" << active.eta << " depth " << active.depth << " thick " 
		<< active.thick << std::endl;
#endif
    }
  }
  return actives;
}

std::vector<HcalCellType> 
HcalDDDRecConstants::HcalCellTypes(HcalSubdetector subdet) const {

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
	std::vector<HcalCellType> cells = hcons.HcalCellTypes(subdet,keta,-1);
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
      int unit = hcons.unitPhi(etabins[bin].dphi);
      for (unsigned int il=0; il<etabins[bin].layer.size(); ++il) {
	int depth = etabins[bin].depthStart + (int)(il);
	temp[il].setEta(ieta,etabins[bin].etaMin,etabins[bin].etaMax);
	temp[il].setPhi(etabins[bin].nPhi,unit,etabins[bin].dphi/CLHEP::deg,
			hpar->phioff[isub]/CLHEP::deg);
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
    return hcons.HcalCellTypes(subdet,-1,-1);
  }
}

unsigned int HcalDDDRecConstants::numberOfCells(HcalSubdetector subdet) const {

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
    return hcons.numberOfCells(subdet);
  }
}

unsigned int HcalDDDRecConstants::nCells(HcalSubdetector subdet) const {

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
  return (nCells(HcalBarrel)+nCells(HcalEndcap)+nCells(HcalOuter)+nCells(HcalForward));
}

void HcalDDDRecConstants::initialize(void) {

  //Eta grouping
  int nEta  = (int)(hpar->etagroup.size());
  if (nEta != (int)(hpar->phigroup.size())) {
    edm::LogError("HCalGeom") << "HcalDDDRecConstants: sizes of the vectors "
                              << " etaGroup (" << nEta << ") and phiGroup ("
                              << hpar->phigroup.size() << ") do not match";
    throw cms::Exception("DDException") << "HcalDDDRecConstants: inconsistent array sizes" << nEta << ":" << hpar->phigroup.size();
  }

  // First eta table
  iEtaMin     = hpar->etaMin;
  iEtaMax     = hpar->etaMax;
  etaTable.clear(); ietaMap.clear(); etaSimValu.clear();
  int ieta(0), ietaHB(0), ietaHE(0);
  etaTable.push_back(hpar->etaTable[ieta]);
  for (int i=0; i<nEta; ++i) {
    int ef = ieta+1;
    ieta  += (hpar->etagroup[i]);
    if (ieta >= (int)(hpar->etaTable.size())) {
      edm::LogError("HCalGeom") << "Going beyond the array boundary "
				<< hpar->etaTable.size() << " at index " << i 
				<< " of etaTable from SimConstant";
      throw cms::Exception("DDException") << "Going beyond the array boundary "
					  << hpar->etaTable.size() 
					  << " at index " << i 
					  << " of etaTable from SimConstant";
    } else {
      etaTable.push_back(hpar->etaTable[ieta]);
      etaSimValu.push_back(std::pair<int,int>(ef,ieta));
    }
    for (int k=0; k<(hpar->etagroup[i]); ++k) ietaMap.push_back(i+1);
    if (ieta <= hpar->etaMax[0]) ietaHB = i+1;
    if (ieta <= hpar->etaMin[1]) ietaHE = i+1;
  }
  iEtaMin[1] = ietaHE;
  iEtaMax[0] = ietaHB;

  // Then Phi bins
  ieta = 0;
  phibin.clear(); phiUnitS.clear();
  for (int i=0; i<nEta; ++i) {
    double dphi = (hpar->phigroup[i])*(hpar->phibin[ieta]);
    phibin.push_back(dphi);
    ieta += (hpar->etagroup[i]);
  }
  for (unsigned int i=1; i<hpar->etaTable.size(); ++i) {
    int unit = hcons.unitPhi(hpar->phibin[i-1]);
    phiUnitS.push_back(unit);
  }

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
  for (unsigned int i=0; i<hpar->etaTableHF.size(); ++i)
    std::cout << " [" << i << "] = " << hpar->etaTableHF[i];
  std::cout << std::endl;
  std::cout << "PhiBinHF";
  for (unsigned int i=0; i<hpar->phitable.size(); ++i)
    std::cout << " [" << i << "] = " << hpar->phitable[i];
  std::cout << std::endl;
#endif

  //Now the depths
  maxDepth    = hpar->maxDepth;
  maxDepth[0] = maxDepth[1] = 0;
  for (int i=0; i<nEta; ++i) {
    unsigned int imx = layerGroupSize(i);
    int laymax = (imx > 0) ? layerGroup(i,imx-1) : 0;
    if (i < iEtaMax[0]) {
      int laymax0 = (imx > 16) ? layerGroup(i,16) : laymax;
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
  nModule[0] = hpar->modHB[0];
  nHalves[0] = hpar->modHB[1];
  for (unsigned int i=0; i<hpar->rHB.size(); ++i) {
    gconsHB.push_back(std::pair<double,double>(hpar->rHB[i]/CLHEP::cm,
					       hpar->drHB[i]/CLHEP::cm));
  }
#ifdef DebugLog
  std::cout << "HB with " << nModule[0] << " modules and " << nHalves[0]
	    <<" halves and " << gconsHB.size() << " layers" << std::endl;
  for (unsigned int i=0; i<gconsHB.size(); ++i) 
    std::cout << "rHB[" << i << "] = " << gconsHB[i].first << " +- "
	      << gconsHB[i].second << std::endl; 
#endif
  nModule[1] = hpar->modHE[0];
  nHalves[1] = hpar->modHE[1];
  for (unsigned int i=0; i<hpar->zHE.size(); ++i) {
    gconsHE.push_back(std::pair<double,double>(hpar->zHE[i]/CLHEP::cm,
					       hpar->dzHE[i]/CLHEP::cm));
  }
#ifdef DebugLog
  std::cout << "HE with " << nModule[1] << " modules and " << nHalves[1] 
	    <<" halves and " << gconsHE.size() << " layers" << std::endl;
  for (unsigned int i=0; i<gconsHE.size(); ++i) 
    std::cout << "zHE[" << i << "] = " << gconsHE[i].first << " +- "
	      << gconsHE[i].second << std::endl; 
#endif
}

unsigned int
HcalDDDRecConstants::layerGroupSize( unsigned int eta ) const
{
  unsigned int k = 0;
  for( auto const & it : hpar->layerGroupEtaRec )
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
HcalDDDRecConstants::layerGroup( unsigned int eta, unsigned int i ) const
{
  unsigned int k = 0;
  for( auto const & it :  hpar->layerGroupEtaRec )
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

const std::vector<int> &
HcalDDDRecConstants::getDepth(const unsigned int i) const
{
  std::vector<HcalParameters::LayerItem>::const_iterator last = hpar->layerGroupEtaRec.begin();
  for( std::vector<HcalParameters::LayerItem>::const_iterator it = hpar->layerGroupEtaRec.begin(); it != hpar->layerGroupEtaRec.end(); ++it ) {
    if( it->layer == i + 1 )
      return it->layerGroup;
    if( it->layer > i + 1 )
      return last->layerGroup;
    last = it;
  }
  return last->layerGroup;
}
