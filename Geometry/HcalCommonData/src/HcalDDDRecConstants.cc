#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

//#define EDM_ML_DEBUG

enum {kHOSizePreLS1 = 2160, kHFSizePreLS1 = 1728} ;

HcalDDDRecConstants::HcalDDDRecConstants(const HcalParameters* hp,
					 const HcalDDDSimConstants& hc) : 
  hpar(hp), hcons(hc) {

#ifdef EDM_ML_DEBUG
  std::cout << "HcalDDDRecConstants::HcalDDDRecConstants (const HcalParameters* hp) constructor" << std::endl;
#endif
  initialize();
}

HcalDDDRecConstants::~HcalDDDRecConstants() { 
#ifdef EDM_ML_DEBUG
  std::cout << "HcalDDDRecConstants::destructed!!!" << std::endl;
#endif
}

std::vector<int> HcalDDDRecConstants::getDepth(const unsigned int& eta,
					       const bool& extra) const {

  if (!extra) {
    std::vector<HcalParameters::LayerItem>::const_iterator last = hpar->layerGroupEtaRec.begin();
    for (std::vector<HcalParameters::LayerItem>::const_iterator it = hpar->layerGroupEtaRec.begin(); it != hpar->layerGroupEtaRec.end(); ++it) {
      if (it->layer == eta + 1) return it->layerGroup;
      if (it->layer > eta + 1 ) return last->layerGroup;
      last = it;
    }
    return last->layerGroup;
  } else {
    std::map<int,int> layers;
    hcons.ldMap()->getLayerDepth(eta+1, layers);
    std::vector<int> depths;
    for (unsigned int lay=0; lay < layers.size(); ++lay) 
      depths.emplace_back(layers[lay+1]);
    return depths;
  }
}

std::vector<int> HcalDDDRecConstants::getDepth(const int& det, const int& phi, const int& zside, 
					       const unsigned int& eta) const {
  std::map<int,int> layers;
  hcons.ldMap()->getLayerDepth(det, eta+1, phi, zside, layers);
  if (layers.empty()) {
    return getDepth(eta, false);
  } else {
    std::vector<int> depths;
    for (unsigned int lay=0; lay < layers.size(); ++lay) 
      depths.emplace_back(layers[lay+1]);
    return depths;
  }
}

std::vector<HcalDDDRecConstants::HcalEtaBin> 
HcalDDDRecConstants::getEtaBins(const int& itype) const {

  std::vector<HcalDDDRecConstants::HcalEtaBin> bins;
  unsigned int     type     = (itype == 0) ? 0 : 1;
  HcalSubdetector  subdet   = HcalSubdetector(type+1);
  std::vector<int> phiSp;
  HcalSubdetector  subdetSp = HcalSubdetector(hcons.ldMap()->validDet(phiSp));
  std::map<int,int> layers;
  for (int iz=0; iz<2; ++iz) {
    int zside = 2*iz - 1;
    for (int ieta = iEtaMin[type]; ieta <= iEtaMax[type]; ++ieta) {
      std::vector<std::pair<int,double> > phis = getPhis(subdet,ieta);
      std::vector<std::pair<int,double> > phiUse;
      getLayerDepth(ieta,layers);
      if (subdet == subdetSp) {
	for (auto & phi : phis) {
	  if (std::find(phiSp.begin(),phiSp.end(),(zside*phi.first)) ==
	      phiSp.end()){
	    phiUse.emplace_back(phi);
	  }
	}
      } else {
	phiUse.insert(phiUse.end(),phis.begin(),phis.end());
      }
      getOneEtaBin(subdet,ieta,zside,phiUse,layers,false,bins);
    }
  }
  if (subdetSp == subdet) {
    for (int ieta = iEtaMin[type]; ieta <= iEtaMax[type]; ++ieta) {
      std::vector<std::pair<int,double> > phis = getPhis(subdet,ieta);
      for (int iz=0; iz<2; ++iz) {
	int zside = 2*iz - 1;
	std::vector<std::pair<int,double> > phiUse;
	for (int i : phiSp) {
	  for (auto & phi : phis) {
	    if (i == zside*phi.first) {
	      phiUse.emplace_back(phi);
	      break;
	    }
	  }
	}
	if (!phiUse.empty()) {
	  hcons.ldMap()->getLayerDepth(subdet,ieta,phiUse[0].first,zside,layers);
	  getOneEtaBin(subdet,ieta,zside,phiUse,layers,true,bins);
	}
      }
    }
  }
#ifdef EDM_ML_DEBUG
  std::cout << "Prepares " << bins.size() << " eta bins for type " << type 
	    << std::endl;
  for (unsigned int i=0; i<bins.size(); ++i) {
    std::cout << "Bin[" << i << "]: Eta = (" << bins[i].ieta << ":"
	      << bins[i].etaMin << ":" << bins[i].etaMax << "), Zside = " 
	      << bins[i].zside << ", phis = (" << bins[i].phis.size() << ":"
	      << bins[i].dphi << ") and " << bins[i].layer.size() 
	      << " depths (start) " << bins[i].depthStart << " :";
    for (unsigned int k=0; k<bins[i].layer.size(); ++k)
      std::cout << " [" << k << "] " << bins[i].layer[k].first << ":"
		<< bins[i].layer[k].second;
    std::cout << std::endl << " and Phi sets";
    for (unsigned int k=0; k<bins[i].phis.size(); ++k)
      std::cout << " " << bins[i].phis[k].first << ":" <<bins[i].phis[k].second;
    std::cout << std::endl;
  }
#endif
  return bins;
}

std::pair<double,double> 
HcalDDDRecConstants::getEtaPhi(const int& subdet, const int& ieta, const int& iphi) const {
  int ietaAbs = (ieta > 0) ? ieta : -ieta;
  double eta(0), phi(0);
  if ((subdet == static_cast<int>(HcalBarrel)) || 
      (subdet == static_cast<int>(HcalEndcap)) ||
      (subdet == static_cast<int>(HcalOuter))) {  // Use Eta Table
    int unit    = hcons.unitPhi(phibin[ietaAbs-1]);
    int kphi    = (unit == 2) ? ((iphi-1)/2 + 1) : iphi;
    double foff = (ietaAbs <= iEtaMax[0]) ? hpar->phioff[0] : hpar->phioff[1];
    eta         = 0.5*(etaTable[ietaAbs-1]+etaTable[ietaAbs]);
    phi         = foff + (kphi-0.5)*phibin[ietaAbs-1];
  } else {
    ietaAbs    -= iEtaMin[2];
    int unit    = hcons.unitPhi(hpar->phitable[ietaAbs-1]);
    int kphi    = (unit == 4) ? ((iphi-3)/4 + 1) : ((iphi-1)/2 + 1);
    double foff = (unit > 2) ? hpar->phioff[4] : hpar->phioff[2];
    eta         = 0.5*(hpar->etaTableHF[ietaAbs-1]+hpar->etaTableHF[ietaAbs]);
    phi         = foff + (kphi-0.5)*hpar->phitable[ietaAbs-1];
  }
  if (ieta < 0)   eta  = -eta;
  if (phi > M_PI) phi -= (2*M_PI);
#ifdef EDM_ML_DEBUG
  std::cout << "getEtaPhi: subdet|ieta|iphi " << subdet << "|" << ieta << "|"
	    << iphi << " eta|phi " << eta << "|" << phi << std::endl;
#endif
  return std::pair<double,double>(eta,phi);
}

HcalDDDRecConstants::HcalID 
HcalDDDRecConstants::getHCID(int subdet, int keta, int iphi, int lay,
			     int idepth) const {

  int    ieta = (keta > 0) ? keta : -keta;
  int    zside= (keta > 0) ? 1 : -1;
  int    eta(ieta), phi(iphi), depth(idepth);
  if ((subdet == static_cast<int>(HcalOuter)) ||
      ((subdet == static_cast<int>(HcalBarrel)) && (lay > maxLayerHB_+1))) {
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
      phi0   = (iphi+1)/4;
      phi0   = (phi0-1)/(hpar->phigroup[eta-1]);
    }
    ++phi0;
    unit     = hcons.unitPhi(phibin[eta-1]);
    phi      = hcons.phiNumber(phi0,unit);
    depth    = hcons.findDepth(subdet,eta,phi,zside,lay-1);
    if (depth <= 0) depth = layerGroup(eta-1, lay-1);
    if (eta == iEtaMin[1]) {
      if (subdet == static_cast<int>(HcalBarrel)) {
	if (depth > hcons.getDepthEta16(subdet,phi,zside))
	  depth = hcons.getDepthEta16(subdet,phi,zside);
      } else {
	if (depth < hcons.getDepthEta16(subdet,phi,zside)) 
	  depth = hcons.getDepthEta16(subdet,phi,zside);
      }
    } else if (eta == hpar->noff[0] && lay > 1) {
      int   kphi   = phi + int((hpar->phioff[3]+0.1)/phibin[eta-1]);
      kphi         = (kphi-1)%4 + 1;
      if (kphi == 2 || kphi == 3) depth = layerGroup(eta-1, lay-2);
    } else if (eta == hpar->noff[1] && 
	       depth > hcons.getDepthEta29(phi,zside,0)) { 
      eta -= hcons.getDepthEta29(phi,zside,1);
    }
  } 
#ifdef EDM_ML_DEBUG
  std::cout << "getHCID: input " << subdet << ":" << ieta << ":" << iphi
	    << ":" << idepth << ":" << lay << " output " << eta << ":" << phi
	    << ":" << depth << std::endl;
#endif
  return HcalDDDRecConstants::HcalID(subdet,eta,phi,depth);
}

std::vector<HcalDDDRecConstants::HFCellParameters> 
HcalDDDRecConstants::getHFCellParameters() const {

  std::vector<HcalDDDRecConstants::HFCellParameters> cells;
  unsigned int nEta = hcons.getPhiTableHF().size();
  if (maxDepth[2] > 0) {
    for (unsigned int k=0; k<nEta; ++k) {
      int ieta = iEtaMin[2] + k;
      int dphi = (int)(0.001 + hcons.getPhiTableHF()[k]/(5.0*CLHEP::deg));
      int iphi = (dphi == 4) ? 3 : 1;
      int nphi = 72/dphi;
      double rMin = hcons.getRTableHF()[nEta-k-1]/CLHEP::cm;
      double rMax = hcons.getRTableHF()[nEta-k]/CLHEP::cm;
      HcalDDDRecConstants::HFCellParameters cell1( ieta,1,iphi,dphi,nphi,rMin,rMax);
      cells.emplace_back(cell1);
      HcalDDDRecConstants::HFCellParameters cell2(-ieta,1,iphi,dphi,nphi,rMin,rMax);
      cells.emplace_back(cell2);
    }
  }
  if (maxDepth[2] > 2) {
    if (!hcons.getIdHF2QIE().empty()) {
      for (unsigned int k=0; k<hcons.getIdHF2QIE().size(); ++k) {
	int ieta = hcons.getIdHF2QIE()[k].ieta();
	int ind  = std::abs(ieta) - iEtaMin[2];
	int dphi = (int)(0.001 + hcons.getPhiTableHF()[ind]/(5.0*CLHEP::deg));
	int iphi = hcons.getIdHF2QIE()[k].iphi();
	double rMin = hcons.getRTableHF()[nEta-ind-1]/CLHEP::cm;
	double rMax = hcons.getRTableHF()[nEta-ind]/CLHEP::cm;
	HcalDDDRecConstants::HFCellParameters cell1( ieta,3,iphi,dphi,1,rMin,rMax);
	cells.emplace_back(cell1);
      }
    } else {
      for (unsigned int k=0; k<nEta; ++k) {
	int ieta = iEtaMin[2] + k;
	int dphi = (int)(0.001 + hcons.getPhiTableHF()[k]/(5.0*CLHEP::deg));
	int iphi = (dphi == 4) ? 3 : 1;
	int nphi = 72/dphi;
	double rMin = hcons.getRTableHF()[nEta-k-1]/CLHEP::cm;
	double rMax = hcons.getRTableHF()[nEta-k]/CLHEP::cm;
	HcalDDDRecConstants::HFCellParameters cell1( ieta,3,iphi,dphi,nphi,rMin,rMax);
	cells.emplace_back(cell1);
	HcalDDDRecConstants::HFCellParameters cell2(-ieta,3,iphi,dphi,nphi,rMin,rMax);
	cells.emplace_back(cell2);
      }
    }
  }
#ifdef EDM_ML_DEBUG
  std::cout << "HcalDDDRecConstants returns " << cells.size() 
	    << " HF cell parameters" << std::endl;
  for (unsigned int k=0; k<cells.size(); ++k)
    std::cout << "Cell[" << k <<"] : (" << cells[k].ieta <<", "<< cells[k].depth
	      << ", " << cells[k].firstPhi << ", " << cells[k].stepPhi << ", "
	      << cells[k].nPhi << ", " << cells[k].rMin << ", "
	      << cells[k].rMax << ")" << std::endl;
#endif
  return cells;
}

void HcalDDDRecConstants::getLayerDepth(const int& ieta, std::map<int,int>& layers) const {

  layers.clear();
  for (unsigned int l=0; l<layerGroupSize(ieta-1); ++l) {
    int lay = l + 1;
    layers[lay] = layerGroup(ieta-1,l);
  }
#ifdef EDM_ML_DEBUG
  std::cout << "getLayerDepth::Input " << ieta << " Output " 
	    << layers.size() << " entries" << std::endl;
  for (std::map<int,int>::iterator itr=layers.begin(); itr != layers.end();
       ++itr) std::cout << " [" << itr->first << "] " << itr->second;
  std::cout << std::endl;
#endif
}

int HcalDDDRecConstants::getLayerFront(const int& idet, const int& ieta,
				       const int& iphi, const int& depth) const {
  int subdet = (idet == 1) ? 1 : 2;
  int zside  = (ieta > 0) ? 1 : -1;
  int eta    = zside*ieta;
  int layFront = hcons.ldMap()->getLayerFront(subdet,eta,iphi,zside,depth);
  if (layFront < 0) {
    int laymin  = hcons.getFrontLayer(subdet, ieta);
    if (eta == 16 && subdet == 2) {
      layFront = laymin;
    } else if (eta <= hpar->etaMax[1]) {
      for (unsigned int k=0; k<layerGroupSize(eta-1); ++k) {
	if (depth == (int)layerGroup(eta-1, k)) {
	  if ((int)(k) >= laymin) {
	    layFront = k;
	    break;
	  }
	}
      }
    }
  }
#ifdef EDM_ML_DEBUG
  std::cout << "getLayerFront::Input " << idet << ":" << ieta << ":"
	    << iphi << ":" << depth << " Output " << layFront << std::endl;
#endif
  return layFront;
}

int HcalDDDRecConstants::getMaxDepth (const int& itype, const int& ieta,
				      const int& iphi,  const int& zside) const {
  
  unsigned int type  = (itype == 0) ? 0 : 1;
  int lmax = hcons.getMaxDepth(type+1, ieta, iphi, zside, true);
  if (lmax < 0) {
    unsigned int lymax = (type == 0) ? maxLayerHB_+1 : maxLayer_+1;
    lmax = 0;
    if (layerGroupSize(ieta-1) > 0) {
      if (layerGroupSize(ieta-1) < lymax) lymax = layerGroupSize(ieta-1);
      lmax = (int)(layerGroup(ieta-1, lymax-1));
      if (type == 0 && ieta == iEtaMax[type]) lmax = hcons.getDepthEta16M(1);
      if (type == 1 && ieta >= hpar->noff[1]) lmax = hcons.getDepthEta29M(0,false);
    }
  }
#ifdef EDM_ML_DEBUG
  std::cout << "getMaxDepth::Input " << itype << ":" << ieta << ":"
	    << iphi << ":" << zside << " Output " << lmax << std::endl;
#endif
  return lmax;
}

int HcalDDDRecConstants::getMinDepth (const int& itype, const int& ieta,
				      const int& iphi,  const int& zside) const {

  int lmin = hcons.getMinDepth(itype+1, ieta, iphi, zside, true);
  if (lmin < 0) {
    if (itype == 2) { // HFn
      lmin = 1;
    } else if (itype == 3) { //HO
      lmin = maxDepth[3];
    } else {
      unsigned int type  = (itype == 0) ? 0 : 1;
      if (layerGroupSize(ieta-1) > 0) {
	if (type == 1 && ieta == iEtaMin[type])
	  lmin = hcons.getDepthEta16M(2);
	else
	  lmin = (int)(layerGroup(ieta-1, 0));
      }
    }
  }
  return lmin;
}

std::vector<std::pair<int,double> >
HcalDDDRecConstants::getPhis(const int& subdet, const int& ieta) const {

  std::vector<std::pair<int,double> > phis;
  int ietaAbs = (ieta > 0) ? ieta : -ieta;
  int    keta  = (subdet != HcalForward) ? etaSimValu[ietaAbs-1].first : ietaAbs;
  std::pair<double,double> ficons = hcons.getPhiCons(subdet, keta);
  double fioff = ficons.first;
  double dphi  = (subdet != HcalForward) ? phibin[ietaAbs-1] : ficons.second;
  int    nphi  = int((CLHEP::twopi+0.1*dphi)/dphi);
  int    units = hcons.unitPhi(subdet, keta);
  for (int ifi = 0; ifi < nphi; ++ifi) {
    double phi =-fioff + (ifi+0.5)*dphi;
    int iphi   = hcons.phiNumber(ifi+1,units);
    phis.emplace_back(std::pair<int,double>(iphi,phi));
  }
#ifdef EDM_ML_DEBUG
  std::cout << "getEtaPhi: subdet|ieta|iphi " << subdet << "|" << ieta 
	    << " with " << phis.size() << " phi bins" << std::endl;
  for (unsigned int k=0; k<phis.size(); ++k)
    std::cout << "[" << k << "] iphi " << phis[k].first << " phi "
	      << phis[k].second/CLHEP::deg << std::endl;
#endif
  return phis;
}

int HcalDDDRecConstants::getPhiZOne(std::vector<std::pair<int,int>>& phiz) const {

  phiz.clear();
  int subdet = hcons.ldMap()->getSubdet();
  if (subdet > 0) {
    std::vector<int> phis = hcons.ldMap()->getPhis();
    for (int k : phis) {
      int zside = (k > 0) ? 1 : -1;
      int phi   = (k > 0) ? k : -k;
      phiz.emplace_back(std::pair<int,int>(phi,zside));
    }
  }
#ifdef EDM_ML_DEBUG
  std::cout << "Special RBX for detector " << subdet << " with " << phiz.size()
	    << " phi/z bins";
  for (unsigned int k=0; k<phiz.size(); ++k)
    std::cout << " [" << k << "] " << phiz[k].first << ":" << phiz[k].second;
  std::cout << std::endl;
#endif
  return subdet;
}

double HcalDDDRecConstants::getRZ(const int& subdet, const int& ieta, 
				  const int& depth) const {

  return getRZ(subdet, ieta, 1, depth);
}

double HcalDDDRecConstants::getRZ(const int& subdet, const int& ieta, const int& iphi,
				  const int& depth) const {
  int    layf  = getLayerFront(subdet,ieta,iphi,depth);
  double rz    = (layf < 0) ? 0.0 : 
    ((subdet == static_cast<int>(HcalBarrel)) ? (gconsHB[layf].first) :
     (gconsHE[layf].first));
#ifdef EDM_ML_DEBUG
  std::cout << "getRZ: subdet|ieta|ipho|depth " << subdet << "|" << ieta << "|"
	    << iphi << "|" << depth << " lay|rz " << layf << "|" << rz 
	    << std::endl;
#endif
  return rz;
}

double HcalDDDRecConstants::getRZ(const int& subdet, const int& layer) const {

  double rz(0);
  if (layer > 0 && layer <= (int)(layerGroupSize(0)))
    rz = ((subdet == static_cast<int>(HcalBarrel)) ? (gconsHB[layer-1].first) :
	  (gconsHE[layer-1].first));
#ifdef EDM_ML_DEBUG
  std::cout << "getRZ: subdet|layer " << subdet << "|" << layer << " rz "
	    << rz << std::endl;
#endif
  return rz;
}

 	
std::vector<HcalDDDRecConstants::HcalActiveLength> 
HcalDDDRecConstants::getThickActive(const int& type) const {

  std::vector<HcalDDDRecConstants::HcalActiveLength> actives;
  std::vector<HcalDDDRecConstants::HcalEtaBin> bins = getEtaBins(type);
#ifdef EDM_ML_DEBUG
  unsigned int kount(0);
#endif
  for (auto & bin : bins) {
    int    ieta  = bin.ieta;
    int    zside = bin.zside;
    int    stype = (bin.phis.size() > 4) ? 0 : 1;
    int    layf  = getLayerFront(type+1,zside*ieta,bin.phis[0].first,bin.depthStart) + 1;
    int    layl  = hcons.getLastLayer(type+1,zside*ieta) + 1;
    double eta   = 0.5*(bin.etaMin+bin.etaMax);
    double theta = 2*atan(exp(-eta));
    double scale = 1.0/((type == 0) ? sin(theta) : cos(theta));
    int    depth = bin.depthStart;
#ifdef EDM_ML_DEBUG
    std::cout << "Eta " << ieta << " zside " << zside << " depth " << depth
	      << " Layers " << layf << ":" << layl << ":" << bin.layer.size();
    for (auto ll : bin.layer) std::cout << " " << ll.first << ":" << ll.second;
    std::cout << " phi ";
    for (auto phi : bin.phis) std::cout << " " << phi.first;
    std::cout << std::endl;
#endif
    for (unsigned int i = 0; i < bin.layer.size(); ++i) {
      double thick(0);
      int lmin = (type == 1 && ieta == iEtaMin[1]) ? layf :
	std::max(bin.layer[i].first,layf);
      int lmax = std::min(bin.layer[i].second,layl);
      for (int j = lmin; j <= lmax; ++j) {
	if (type == 0 || j > 1) {
	  double t = ((type == 0) ? gconsHB[j-1].second : gconsHE[j-1].second);
	  if (t > 0) thick += t;
	}
      }
      thick *= (2.*scale);
      HcalDDDRecConstants::HcalActiveLength active(ieta,depth,zside,stype,zside*eta,thick);
      for (auto phi : bin.phis) 
	active.iphis.emplace_back(phi.first);
      actives.emplace_back(active);
      ++depth;
#ifdef EDM_ML_DEBUG
      kount++;
      std::cout << "getThickActive: [" << kount << "] eta:" << active.ieta 
		<< ":" << active.eta << " zside " << active.zside << " depth " 
		<< active.depth << " type " << active.stype << " thick "
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
    std::vector<int> missPhi;
    for (const auto& etabin : etabins) {
      std::vector<HcalCellType> temp;
      std::vector<int>          count;
      std::vector<double>       dmin, dmax;
      for (unsigned int il=0; il<etabin.layer.size(); ++il) {
	HcalCellType cell(subdet, etabin.ieta, etabin.zside, 0,
			  HcalCellType::HcalCell());
	temp.emplace_back(cell);
	count.emplace_back(0);
	dmin.emplace_back(0);
	dmax.emplace_back(0);
      }
      int ieta = etabin.ieta;
      for (int keta=etaSimValu[ieta-1].first; keta<=etaSimValu[ieta-1].second;
	   ++keta) {
	std::vector<HcalCellType> cellsm = hcons.HcalCellTypes(subdet,keta,-1);
	for (unsigned int il=0; il<etabin.layer.size(); ++il) {
	  for (auto & ic : cellsm) {
	    if (ic.depthSegment() >= etabin.layer[il].first &&
		ic.depthSegment() <= etabin.layer[il].second &&
		ic.etaBin() == temp[il].etaBin() && 
		ic.zside()  == temp[il].zside()) {
	      if (count[il] == 0) {
		temp[il] = ic;
		dmin[il] = ic.depthMin();
		dmax[il] = ic.depthMax();
	      }
	      ++count[il];
	      if (ic.depthMin() < dmin[il]) 
		dmin[il] = ic.depthMin();
	      if (ic.depthMax() > dmax[il]) 
		dmax[il] = ic.depthMax();
	    }
	  }
	}
      }
      for (unsigned int il=0; il<etabin.layer.size(); ++il) {
	int depth = etabin.depthStart + (int)(il);
	temp[il].setEta(ieta,etabin.etaMin,etabin.etaMax);
	temp[il].setDepth(depth,dmin[il],dmax[il]);
	double foff = (etabin.ieta <= iEtaMax[0]) ? hpar->phioff[0] : hpar->phioff[1];
	int unit    = hcons.unitPhi(etabin.dphi);
	temp[il].setPhi(etabin.phis, missPhi, foff, etabin.dphi, unit);
	cells.emplace_back(temp[il]);
      }
    }
#ifdef EDM_ML_DEBUG
    std::cout << "HcalDDDRecConstants: found " << cells.size() 
	      << " cells for sub-detector type " << isub << std::endl;
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
    for (auto & cellType : cellTypes) {
      num += (unsigned int)(cellType.nPhiBins());
    }
#ifdef EDM_ML_DEBUG
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
    for (auto & etabin : etabins) {
      ncell += ((etabin.phis.size())*(etabin.layer.size()));
    }
    return ncell;
  } else if (subdet == HcalOuter) {
    return kHOSizePreLS1;
  } else if (subdet == HcalForward) {
    return (unsigned int)(hcons.numberOfCells(subdet));
  } else {
    return 0;
  }
}

unsigned int HcalDDDRecConstants::nCells() const {
  return (nCells(HcalBarrel)+nCells(HcalEndcap)+nCells(HcalOuter)+nCells(HcalForward));
}

HcalDetId HcalDDDRecConstants::mergedDepthDetId(const HcalDetId& id) const {

  std::map<HcalDetId,HcalDetId>::const_iterator itr = detIdSp_.find(id);
  if (itr == detIdSp_.end()) return id;
  else                       return itr->second;
}

HcalDetId HcalDDDRecConstants::idFront(const HcalDetId& id) const {

  HcalDetId hid(id);
  std::map<HcalDetId,std::vector<HcalDetId>>::const_iterator itr = detIdSpR_.find(id);
  if (itr != detIdSpR_.end())
    hid = HcalDetId(id.subdet(),id.ieta(),id.iphi(),(itr->second)[0].depth());
  return hid;
}

HcalDetId HcalDDDRecConstants::idBack(const HcalDetId& id) const {

  HcalDetId hid(id);
  std::map<HcalDetId,std::vector<HcalDetId>>::const_iterator itr = detIdSpR_.find(id);
  if (itr != detIdSpR_.end())
    hid = HcalDetId(id.subdet(),id.ieta(),id.iphi(),(itr->second).back().depth());
  return hid;
}

void HcalDDDRecConstants::unmergeDepthDetId(const HcalDetId& id,
					    std::vector<HcalDetId>& ids) const {

  ids.clear();
  std::map<HcalDetId,std::vector<HcalDetId>>::const_iterator itr = detIdSpR_.find(id);
  if (itr == detIdSpR_.end()) {
    ids.emplace_back(id);
  } else {
    for (auto k : itr->second) {
      HcalDetId hid(id.subdet(),id.ieta(),id.iphi(),k.depth());
      ids.emplace_back(hid);
    }
  }
}

void HcalDDDRecConstants::specialRBXHBHE(const std::vector<HcalDetId>& idsOld,
					 std::vector<HcalDetId>& idsNew) const {
  for (auto k : idsOld) {
    std::map<HcalDetId,HcalDetId>::const_iterator itr = detIdSp_.find(k);
    if (itr == detIdSp_.end()) idsNew.emplace_back(k);
    else                       idsNew.emplace_back(itr->second);
  }
}

bool HcalDDDRecConstants::specialRBXHBHE(bool tobemerged,
					 std::vector<HcalDetId>& ids) const {
  if (tobemerged) {
    std::map<HcalDetId,HcalDetId>::const_iterator itr;
    for (itr = detIdSp_.begin(); itr != detIdSp_.end(); ++itr) 
      ids.emplace_back(itr->first);
  } else{
    std::map<HcalDetId,std::vector<HcalDetId>>::const_iterator itr;
    for (itr = detIdSpR_.begin(); itr != detIdSpR_.end(); ++itr) 
      ids.emplace_back(itr->first);
  }
  return (!ids.empty());
}

void HcalDDDRecConstants::getOneEtaBin(HcalSubdetector subdet, int ieta, int zside,
				       std::vector<std::pair<int,double> >& phis,
				       std::map<int,int>& layers, bool planOne,
				       std::vector<HcalDDDRecConstants::HcalEtaBin>& bins) const {

  unsigned int lymax = (subdet == HcalBarrel) ? maxLayerHB_+1 : maxLayer_+1;
  int          type  = (subdet == HcalBarrel) ? 0 : 1;
  double       dphi  = phibin[ieta-1];
  HcalDDDRecConstants::HcalEtaBin etabin = HcalDDDRecConstants::HcalEtaBin(ieta,zside,dphi,etaTable[ieta-1],etaTable[ieta]);
  etabin.phis.insert(etabin.phis.end(),phis.begin(),phis.end());
  int n = (ieta == iEtaMax[type]) ? 0 : 1;
  HcalDDDRecConstants::HcalEtaBin etabin0= HcalDDDRecConstants::HcalEtaBin(ieta,zside,dphi,etaTable[ieta-1],etaTable[ieta+n]);
  etabin0.depthStart = hcons.getDepthEta29(phis[0].first,zside,0)+1;
  int dstart = -1;
  int lmin(0), lmax(0);
  std::map<int,int>::iterator itr=layers.begin();
  if (!layers.empty()) {
    int dep = itr->second;
    if (subdet == HcalEndcap && ieta == iEtaMin[type]) 
      dep = hcons.getDepthEta16(subdet,phis[0].first,zside);
    unsigned lymx0 = (layers.size() > lymax) ? lymax : layers.size();
#ifdef EDM_ML_DEBUG
    std::cout << "Eta " << ieta << ":" << hpar->noff[1] << " zside " << zside
	      << " lymax " << lymx0 << ":" << lymax << " Depth " << dep << ":"
	      << itr->second;
    unsigned int l(0);
    for (itr = layers.begin(); itr != layers.end(); ++itr,++l)
      std::cout << " [" << l << "] " << itr->first << ":" << itr->second;
    std::cout  << std::endl << " with " << phis.size() << " phis";
    for (unsigned int l=0; l<phis.size(); ++l)
      std::cout << " " << phis[l].first << ":" << phis[l].second;
    std::cout << std::endl;
#endif
    for (itr = layers.begin(); itr != layers.end(); ++itr) {
      if (itr->first <= (int)(lymx0)) {
	if (itr->second == dep) {
	  if (lmin == 0) lmin = itr->first;
	  lmax = itr->first;
	} else if (itr->second > dep) {
	  if (dstart < 0) dstart = dep;
	  int lmax0 = (lmax >= lmin) ? lmax : lmin;
	  if (subdet == HcalEndcap && ieta+1 == hpar->noff[1] && 
	      dep > hcons.getDepthEta29(phis[0].first,zside,0)) {
	    etabin0.layer.emplace_back(std::pair<int,int>(lmin,lmax0));
	  } else {
	    etabin.layer.emplace_back(std::pair<int,int>(lmin,lmax0));
	  }
	  lmin = itr->first;
	  lmax = lmin-1;
	  dep  = itr->second;
	}
	if (subdet == HcalBarrel && ieta == iEtaMax[type] && 
	    dep > hcons.getDepthEta16M(1)) break;
	if (subdet == HcalEndcap && ieta == hpar->noff[1] &&
	    dep > hcons.getDepthEta29M(0,planOne)) {
	  lmax = lymx0;
	  break;
	}
	if (itr->first == (int)(lymx0)) lmax = lymx0;
      }
    }
    if (lmax >= lmin) {
      if (ieta+1 == hpar->noff[1]) {
	etabin0.layer.emplace_back(std::pair<int,int>(lmin,lmax));
	etabin0.phis.insert(etabin0.phis.end(),phis.begin(),phis.end());
	bins.emplace_back(etabin0);
#ifdef EDM_ML_DEBUG
	std::cout << "etabin0: dStatrt " << etabin0.depthStart << " layers "
		  << etabin0.layer.size() << ":" << lmin << ":" << lmax
		  << " phis " << phis.size() << std::endl;
	for (unsigned int k=0; k<etabin0.layer.size(); ++k)
	  std::cout << " [" << k << "] " << etabin0.layer[k].first << ":"
		    << etabin0.layer[k].second;
	std::cout << std::endl;
#endif
      } else if (ieta == hpar->noff[1]) {
      } else {
	etabin.layer.emplace_back(std::pair<int,int>(lmin,lmax));
	if (dstart < 0) dstart = dep;
      }
    }
  }
  etabin.depthStart = dstart;
  bins.emplace_back(etabin);
#ifdef EDM_ML_DEBUG
  std::cout << "etabin: dStatrt " << etabin.depthStart << " layers "
	    << etabin.layer.size() << ":" << lmin << ":" << lmax
	    << " phis " << etabin.phis.size() << std::endl;
  for (unsigned int k=0; k<etabin.layer.size(); ++k)
    std::cout << " [" << k << "] " << etabin.layer[k].first << ":"
	      << etabin.layer[k].second;
  std::cout << std::endl;
#endif
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
  int ieta(0), ietaHB(0), ietaHE(0), ietaHEM(0);
  etaTable.emplace_back(hpar->etaTable[ieta]);
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
      etaTable.emplace_back(hpar->etaTable[ieta]);
      etaSimValu.emplace_back(std::pair<int,int>(ef,ieta));
    }
    for (int k=0; k<(hpar->etagroup[i]); ++k) ietaMap.emplace_back(i+1);
    if (ieta <= hpar->etaMax[0]) ietaHB = i+1;
    if (ieta <= hpar->etaMin[1]) ietaHE = i+1;
    if (ieta <= hpar->etaMax[1]) ietaHEM= i+1;
  }
  iEtaMin[1] = ietaHE;
  iEtaMax[0] = ietaHB;
  iEtaMax[1] = ietaHEM;

  // Then Phi bins
  nPhiBins.clear();
  for (unsigned int k=0; k<4; ++k) nPhiBins.emplace_back(0);
  ieta = 0;
  phibin.clear(); phiUnitS.clear();
  for (int i=0; i<nEta; ++i) {
    double dphi = (hpar->phigroup[i])*(hpar->phibin[ieta]);
    phibin.emplace_back(dphi);
    int    nphi = (int)((CLHEP::twopi + 0.001)/dphi);
    if (ieta <= iEtaMax[0]) {
      if (nphi > nPhiBins[0]) nPhiBins[3] = nPhiBins[0] = nphi;
    }
    if (ieta >= iEtaMin[1]) {
      if (nphi > nPhiBins[1]) nPhiBins[1] = nphi;
    }
    ieta += (hpar->etagroup[i]);
  }
  for (unsigned int i=1; i<hpar->etaTable.size(); ++i) {
    int unit = hcons.unitPhi(hpar->phibin[i-1]);
    phiUnitS.emplace_back(unit);
  }
  for (double i : hpar->phitable)  {
    int  nphi = (int)((CLHEP::twopi + 0.001)/i);
    if (nphi > nPhiBins[2]) nPhiBins[2] = nphi;
  }
#ifdef EDM_ML_DEBUG
  std::cout << "Modified eta/deltaphi table for " << nEta << " bins" << std::endl;
  for (int i=0; i<nEta; ++i)
    std::cout << "Eta[" << i << "] = " << etaTable[i] << ":" << etaTable[i+1]
	      << ":" << etaSimValu[i].first << ":" << etaSimValu[i].second
	      << " PhiBin[" << i << "] = " << phibin[i]/CLHEP::deg <<std::endl;
  std::cout << "PhiUnitS";
  for (unsigned int i=0; i<phiUnitS.size(); ++i)
    std::cout << " [" << i << "] = " << phiUnitS[i];
  std::cout << std::endl;
  std::cout << "nPhiBins";
  for (unsigned int i=0; i<nPhiBins.size(); ++i)
    std::cout << " [" << i << "] = " << nPhiBins[i];
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
      if (i+1 == iEtaMax[0]) laymax0 = hcons.getDepthEta16M(1);
#ifdef EDM_ML_DEBUG
      std::cout << "HB " << i << " " << imx << " " << laymax << " " 
		<< laymax0 << std::endl;
#endif
      if (maxDepth[0] < laymax0) maxDepth[0] = laymax0;
    }
    if (i >= iEtaMin[1]-1 && i < iEtaMax[1]) {
#ifdef EDM_ML_DEBUG
      std::cout << "HE " << i << " " << imx << " " << laymax << std::endl;
#endif
      if (maxDepth[1] < laymax) maxDepth[1] = laymax;
    }
  }
#ifdef EDM_ML_DEBUG
  for (int i=0; i<4; ++i) 
    std::cout << "Detector Type[" << i << "] iEta " << iEtaMin[i] << ":"
              << iEtaMax[i] << " MaxDepth " << maxDepth[i] << std::endl; 
#endif

  //Now the geometry constants
  nModule[0] = hpar->modHB[0];
  nHalves[0] = hpar->modHB[1];
  for (unsigned int i=0; i<hpar->rHB.size(); ++i) {
    gconsHB.emplace_back(std::pair<double,double>(hpar->rHB[i]/CLHEP::cm,
					       hpar->drHB[i]/CLHEP::cm));
  }
#ifdef EDM_ML_DEBUG
  std::cout << "HB with " << nModule[0] << " modules and " << nHalves[0]
	    <<" halves and " << gconsHB.size() << " layers" << std::endl;
  for (unsigned int i=0; i<gconsHB.size(); ++i) 
    std::cout << "rHB[" << i << "] = " << gconsHB[i].first << " +- "
	      << gconsHB[i].second << std::endl; 
#endif
  nModule[1] = hpar->modHE[0];
  nHalves[1] = hpar->modHE[1];
  for (unsigned int i=0; i<hpar->zHE.size(); ++i) {
    gconsHE.emplace_back(std::pair<double,double>(hpar->zHE[i]/CLHEP::cm,
					       hpar->dzHE[i]/CLHEP::cm));
  }
#ifdef EDM_ML_DEBUG
  std::cout << "HE with " << nModule[1] << " modules and " << nHalves[1] 
	    <<" halves and " << gconsHE.size() << " layers" << std::endl;
  for (unsigned int i=0; i<gconsHE.size(); ++i) 
    std::cout << "zHE[" << i << "] = " << gconsHE[i].first << " +- "
	      << gconsHE[i].second << std::endl; 
#endif

  //Special RBX
  depthMaxSp_ = hcons.getMaxDepthDet(0);
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

  //Map of special DetId's
  std::vector<int> phis;
  HcalSubdetector subdet = HcalSubdetector(hcons.ldMap()->validDet(phis));
  detIdSp_.clear(); detIdSpR_.clear();
  if ((subdet == HcalBarrel) || (subdet == HcalEndcap)) {
    int             phi    = (phis[0] > 0) ? phis[0] : -phis[0];
    int             zside  = (phis[0] > 0) ? 1 : -1;
    int lymax = (subdet == HcalBarrel) ? maxLayerHB_+1 : maxLayer_+1;
    std::pair<int,int>etas = hcons.ldMap()->validEta();
    for (int eta=etas.first; eta<=etas.second; ++eta) {
      std::map<int,std::pair<int,int> > oldDep;
      int depth(0);
      int lmin = layerGroup(eta-1,0);
      for (int lay=0; lay<lymax; ++lay) {
	int depc = layerGroup(eta-1,lay);
	if (depth != depc) {
	  if (depth != 0) oldDep[depth] = std::pair<int,int>(lmin,lay-1);
	  depth = depc;
	  lmin  = lay;
	}
      }
      if (depth != 0) oldDep[depth] = std::pair<int,int>(lmin,lymax-1);
#ifdef EDM_ML_DEBUG      
      std::cout << "Eta|Phi|Zside " << eta << ":" << phi << ":" << zside
		<< " with " << oldDep.size() << " old Depths" << std::endl;
      unsigned int kk(0);
      for (std::map<int,std::pair<int,int> >::const_iterator itr=oldDep.begin(); itr != oldDep.end(); ++itr,++kk)
	std::cout << "[" << kk << "] " << itr->first << " --> " 
		  << itr->second.first << ":" << itr->second.second << "\n";
#endif
      std::pair<int,int> depths = hcons.ldMap()->getDepths(eta);
      for (int ndepth=depths.first; ndepth<=depths.second; ++ndepth) {
	bool flag = ((subdet == HcalBarrel && eta == iEtaMax[0] && 
		      ndepth > hcons.getDepthEta16(subdet,phi,zside)) ||
		     (subdet == HcalEndcap && eta == iEtaMin[1] &&
		      ndepth < hcons.getDepthEta16(subdet,phi,zside)));
	if (!flag) {
	  std::vector<int> count(oldDep.size(),0);
	  int layFront = hcons.ldMap()->getLayerFront(subdet,eta,phi,zside,ndepth);
	  int layBack  = hcons.ldMap()->getLayerBack(subdet,eta,phi,zside,ndepth);
	  for (int lay=layFront; lay<=layBack; ++lay) {
	    unsigned int l(0);
	    for (std::map<int,std::pair<int,int> >::iterator itr=oldDep.begin();
		 itr != oldDep.end(); ++itr,++l) {
	      if (lay >= (itr->second).first && lay <= (itr->second).second) {
		++count[l]; break;
	      }
	    }
	  }
	  int odepth(0), maxlay(0);
	  unsigned int l(0);
	  for (std::map<int,std::pair<int,int> >::iterator itr=oldDep.begin();
	       itr != oldDep.end(); ++itr,++l) {
	    if (count[l] > maxlay) {
	      odepth = itr->first;
	      maxlay = count[l];
	    }
	  }
#ifdef EDM_ML_DEBUG      
	  std::cout << "New Depth " << ndepth << " old Depth " << odepth 
		    << " max " << maxlay << std::endl;
#endif
	  for (int k : phis) {
	    zside  = (k > 0) ? 1 : -1;
	    phi    = (k > 0) ? k : -k;
	    if (subdet == HcalEndcap && eta == hpar->noff[1] &&
		ndepth > hcons.getDepthEta29M(0,true)) break;
	    HcalDetId newId(subdet,zside*eta,phi,ndepth);
	    HcalDetId oldId(subdet,zside*eta,phi,odepth);
	    detIdSp_[newId] = oldId;
	    std::vector<HcalDetId> ids;
	    std::map<HcalDetId,std::vector<HcalDetId>>::iterator itr = detIdSpR_.find(oldId);
	    if (itr != detIdSpR_.end()) ids = itr->second;
	    ids.emplace_back(newId);
	    detIdSpR_[oldId] = ids;
	  }
	}
      }
    }
#ifdef EDM_ML_DEBUG
    std::cout << "Map for merging new channels to old channel IDs with "
	      << detIdSp_.size() << " entries" << std::endl;
    int l(0);
    for (auto itr : detIdSp_) {
      std::cout << "[" << l << "] Special " << itr.first << " Standard "
		<< itr.second << std::endl;
      ++l;
    }
    std::cout << "Reverse Map for mapping old to new IDs with "
	      << detIdSpR_.size() << " entries" << std::endl;
    l = 0;
    for (auto itr : detIdSpR_) {
      std::cout << "[" << l << "] Standard " << itr.first << " Special";
      for (auto itr1 : itr.second) 
	std::cout << " " << (itr1);
      std::cout << std::endl;
      ++l;
    }
#endif
  }

}

unsigned int HcalDDDRecConstants::layerGroupSize(int eta) const {
  unsigned int k = 0;
  for (auto const & it : hpar->layerGroupEtaRec) {
    if (it.layer == (unsigned int)(eta + 1)) {
      return it.layerGroup.size();
    }
    if (it.layer > (unsigned int)(eta + 1)) break;
    k = it.layerGroup.size();
  }
  return k;
}

unsigned int HcalDDDRecConstants::layerGroup(int eta, int i) const {
  unsigned int k = 0;
  for (auto const & it :  hpar->layerGroupEtaRec) {
    if (it.layer == (unsigned int)(eta + 1))  {
      return it.layerGroup.at(i);
    }
    if (it.layer > (unsigned int)(eta + 1)) break;
    k = it.layerGroup.at(i);
  }
  return k;
}
