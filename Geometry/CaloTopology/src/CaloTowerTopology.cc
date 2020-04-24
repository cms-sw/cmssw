#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"
#include <cassert>
#include <algorithm>

//#define DebugLog

CaloTowerTopology::CaloTowerTopology(const HcalTopology * topology) : hcaltopo(topology) {

  //get number of towers in each hcal subdet from hcaltopo
  int nEtaHB, nEtaHO, nEtaHF;
  nEtaHB = hcaltopo->lastHBRing() - hcaltopo->firstHBRing() + 1;
  nEtaHE_= hcaltopo->lastHERing() - hcaltopo->firstHERing() + 1;
  nEtaHO = hcaltopo->lastHORing() - hcaltopo->firstHORing() + 1;
  nEtaHF = hcaltopo->lastHFRing() - hcaltopo->firstHFRing() + 1;
#ifdef DebugLog
  std::cout << "CaloTowerTopology:(1) " << nEtaHB << ":" << nEtaHE_ << ":"
	    << nEtaHO << ":" << nEtaHF << ":" << hcaltopo->isBH() << std::endl;
#endif
  if (hcaltopo->isBH()) nEtaHE_ = 0;

  //setup continuous ieta  
  firstHBRing_ = 1;
  lastHBRing_ = firstHBRing_ + nEtaHB - 1;
  firstHERing_ = lastHBRing_; //crossover
  lastHERing_ = firstHERing_ + std::max(nEtaHE_ - 1,0); //max = protection for case with no HE
  firstHFRing_ = lastHERing_ + 1; //no crossover for CaloTowers; HF crossover cells go in the subsequent non-crossover HF tower
  lastHFRing_ = firstHFRing_ + (nEtaHF - 1) - 1; //nEtaHF - 1 to account for no crossover
  firstHORing_ = 1;
  lastHORing_ = firstHORing_ + nEtaHO - 1;
#ifdef DebugLog
  std::cout << "CaloTowerTopology: (2) " << firstHBRing_ << ":" << lastHBRing_
	    << ":" << firstHERing_ << ":" << lastHERing_ << ":" << firstHFRing_
	    << ":" << lastHFRing_ << ":" << firstHORing_ << ":" << lastHORing_
	    << std::endl;
#endif

  //translate phi segmentation boundaries into continuous ieta
  if(hcaltopo->firstHEDoublePhiRing()==999 || nEtaHE_ == 0) firstHEDoublePhiRing_ = firstHFRing_;
  else firstHEDoublePhiRing_ = firstHERing_ + (hcaltopo->firstHEDoublePhiRing() - hcaltopo->firstHERing());
  firstHEQuadPhiRing_ = firstHERing_ + (hcaltopo->firstHEQuadPhiRing() - hcaltopo->firstHERing());
  firstHFQuadPhiRing_ = firstHFRing_ - 1 + (hcaltopo->firstHFQuadPhiRing() - hcaltopo->firstHFRing());
  
  //number of etas per phi segmentation type
  int nEtaSinglePhi_, nEtaDoublePhi_, nEtaQuadPhi_;
  nEtaSinglePhi_ = firstHEDoublePhiRing_ - firstHBRing_;
  nEtaDoublePhi_ = firstHFQuadPhiRing_ - firstHEDoublePhiRing_;
  nEtaQuadPhi_ = lastHFRing_ - firstHFQuadPhiRing_ + 1; //include lastHFRing
  
  //total number of towers per phi segmentation type
  nSinglePhi_ = nEtaSinglePhi_*72;
  nDoublePhi_ = nEtaDoublePhi_*36;
  nQuadPhi_ = nEtaQuadPhi_*18;

#ifdef DebugLog
  std::cout << "CaloTowerTopology: (3) " << nEtaSinglePhi_ << ":" 
	    << nEtaDoublePhi_ << ":" << nEtaQuadPhi_ << ":" << nSinglePhi_
	    << ":" << nDoublePhi_ << ":" << nQuadPhi_ << std::endl;
#endif
  
  //calculate maximum dense index size
  kSizeForDenseIndexing = 2*(nSinglePhi_ + nDoublePhi_ + nQuadPhi_);

}

//convert CaloTowerTopology ieta to HcalTopology ieta
int CaloTowerTopology::convertCTtoHcal(int ct_ieta) const {
  if(ct_ieta <= lastHBRing_) return ct_ieta - firstHBRing_ + hcaltopo->firstHBRing();
  else if(ct_ieta <= lastHERing_) return ct_ieta - firstHERing_ + hcaltopo->firstHERing();
  else if(ct_ieta <= lastHFRing_) return ct_ieta - firstHFRing_ + hcaltopo->firstHFRing() + 1; //account for no HF crossover
  else return 0; //if ct_ieta outside range
}

//convert HcalTopology ieta to CaloTowerTopology ieta
int CaloTowerTopology::convertHcaltoCT(int hcal_ieta, HcalSubdetector subdet) const {
  if(subdet == HcalBarrel && hcal_ieta >= hcaltopo->firstHBRing() && hcal_ieta <= hcaltopo->lastHBRing()){
    return hcal_ieta - hcaltopo->firstHBRing() + firstHBRing_;
  }
  else if(subdet == HcalEndcap && hcal_ieta >= hcaltopo->firstHERing() && hcal_ieta <= hcaltopo->lastHERing()){
    return ((nEtaHE_ == 0) ? 0 : (hcal_ieta - hcaltopo->firstHERing() + firstHERing_));
  }
  else if(subdet == HcalForward && hcal_ieta >= hcaltopo->firstHFRing() && hcal_ieta <= hcaltopo->lastHFRing()) {
  	if(hcal_ieta == hcaltopo->firstHFRing()) hcal_ieta++; //account for no HF crossover
  	return hcal_ieta - hcaltopo->firstHFRing() + firstHFRing_ - 1;
  }
  else if(subdet == HcalOuter && hcal_ieta >= hcaltopo->firstHORing() && hcal_ieta <= hcaltopo->lastHORing()) {
    return hcal_ieta - hcaltopo->firstHORing() + firstHORing_;
  }
  else return 0; //if hcal_ieta outside range, or unknown subdet
}

bool CaloTowerTopology::valid(const DetId& id) const {
  assert(id.det()==DetId::Calo && id.subdetId()==CaloTowerDetId::SubdetId);
  return validDetId(id);
}

bool CaloTowerTopology::validDetId(const CaloTowerDetId& id) const {
  int ia = id.ietaAbs();
  int ip = id.iphi();
  
  return ( (ia >= firstHBRing_) && (ia <= lastHFRing_) //eta range
           && (ip >= 1) && (ip <= 72) //phi range
		   && (   (ia < firstHEDoublePhiRing_) //72 phi segments
               || (ia < firstHFQuadPhiRing_ && (ip-1)%2 == 0) //36 phi segments, numbered 1,3,...,33,35
               || (ia >= firstHFQuadPhiRing_ && (ip-3)%4 == 0)  ) //18 phi segments, numbered 71,3,7,11,...
         );
}

//decreasing ieta
std::vector<DetId> CaloTowerTopology::east(const DetId& id) const {
  std::vector<DetId> dd;
  CaloTowerDetId tid(id);
  int ieta=tid.ieta();
  int iphi=tid.iphi();
  
  if (ieta==1) { //no ieta=0
    ieta=-1;
  } else if (ieta==firstHEDoublePhiRing_) { //currently double phi, going to single phi (positive eta) -> extra neighbor
    ieta--;
    dd.emplace_back(CaloTowerDetId(ieta,iphi+1));    
  } else if (ieta-1==-firstHEDoublePhiRing_) { //currently single phi, going to double phi (negative eta) -> change numbering
    if ((iphi%2)==0) iphi--;
    ieta--;
  } else if (ieta==firstHFQuadPhiRing_) { //currently quad phi, going to double phi (positive eta) -> extra neighbor
    ieta--;
    dd.emplace_back(CaloTowerDetId(ieta,((iphi+1)%72)+1));    
  } else if (ieta-1==-firstHFQuadPhiRing_) { //currently double phi, going to quad phi (negative eta) -> change numbering
    if (((iphi-1)%4)==0) {
      if (iphi==1) iphi=71;
      else         iphi-=2;
    }
    ieta--;
  } else { //general case
    ieta--;
  }

  if (ieta>=-lastHFRing_) dd.emplace_back(CaloTowerDetId(ieta,iphi));
  return dd;
}

//increasing ieta
std::vector<DetId> CaloTowerTopology::west(const DetId& id) const {
  std::vector<DetId> dd;
  CaloTowerDetId tid(id);

  int ieta=tid.ieta();
  int iphi=tid.iphi();

  if (ieta==-1) { //no ieta=0
    ieta=1;
  } else if (ieta==-firstHEDoublePhiRing_) { //currently double phi, going to single phi (negative eta) -> extra neighbor
    ieta++;
    dd.emplace_back(CaloTowerDetId(ieta,iphi+1));    
  } else if (ieta+1==firstHEDoublePhiRing_) { //currently single phi, going to double phi (positive eta) -> change numbering
    if ((iphi%2)==0) iphi--;
    ieta++;
  } else if (ieta==-firstHFQuadPhiRing_) { //currently quad phi, going to double phi (negative eta) -> extra neighbor
    ieta++;
    dd.emplace_back(CaloTowerDetId(ieta,((iphi+1)%72)+1));
  } else if (ieta+1==firstHFQuadPhiRing_) { //currently double phi, going to quad phi (positive eta) -> change numbering
    if (((iphi-1)%4)==0) {
      if (iphi==1) iphi=71;
      else         iphi-=2;
    }
    ieta++;
  } else {
    ieta++;
  }

  if (ieta<=lastHFRing_) dd.emplace_back(CaloTowerDetId(ieta,iphi));

  return dd;
}

//increasing iphi
std::vector<DetId> CaloTowerTopology::north(const DetId& id) const {
  CaloTowerDetId tid(id);
  int iphi_n=tid.iphi()+1;
  if (iphi_n>72) iphi_n=1;
  if (tid.ietaAbs()>=firstHFQuadPhiRing_) { //18 phi segments, numbered 71,3,7,11,...
    iphi_n+=3;
    if (iphi_n>72) iphi_n-=72;
  } else if (tid.ietaAbs()>=firstHEDoublePhiRing_ && (iphi_n%2)==0) { //36 phi segments, numbered 1,3,...,33,35
    iphi_n++;
    if (iphi_n>72) iphi_n-=72;
  }

  std::vector<DetId> dd;
  dd.emplace_back(CaloTowerDetId(tid.ieta(),iphi_n));
  return dd;
}

//decreasing iphi
std::vector<DetId> CaloTowerTopology::south(const DetId& id) const {
  CaloTowerDetId tid(id);
  int iphi_s=tid.iphi()-1;
  if (iphi_s==0) iphi_s=72;
  if (tid.ietaAbs()>=firstHFQuadPhiRing_) { //18 phi segments, numbered 71,3,7,11,...
    iphi_s-=3;
    if (iphi_s<=0) iphi_s+=72;
  } else if (tid.ietaAbs()>=firstHEDoublePhiRing_ && (iphi_s%2)==0) { //36 phi segments, numbered 1,3,...,33,35
    iphi_s--;
  }

  std::vector<DetId> dd;
  dd.emplace_back(CaloTowerDetId(tid.ieta(),iphi_s));
  return dd;
}

std::vector<DetId> CaloTowerTopology::up(const DetId& /*id*/) const {
  return std::vector<DetId>();
}

std::vector<DetId> CaloTowerTopology::down(const DetId& /*id*/) const {
  return std::vector<DetId>();
}

uint32_t CaloTowerTopology::denseIndex(const DetId& id) const {
  CaloTowerDetId tid(id);
  const int ie ( tid.ietaAbs() );
  const int ip ( tid.iphi() - 1 ) ;
  
  return ( ( 0 > tid.zside() ? 0 : kSizeForDenseIndexing/2 ) +
           ( ( firstHEDoublePhiRing_ > ie ? ( ie - 1 )*72 + ip :
	       ( firstHFQuadPhiRing_ > ie ?  nSinglePhi_ + ( ie - firstHEDoublePhiRing_ )*36 + ip/2 :
		 nSinglePhi_ + nDoublePhi_ + ( ie - firstHFQuadPhiRing_ )*18 + ip/4 ) ) ) );
}

CaloTowerDetId CaloTowerTopology::detIdFromDenseIndex( uint32_t din ) const {
  const int iz ( din < kSizeForDenseIndexing/2 ? -1 : 1 ) ;
  din %= kSizeForDenseIndexing/2 ;
  const int ie ( nSinglePhi_ + nDoublePhi_ - 1 < (int)(din) ?
		 firstHFQuadPhiRing_ + (din - nSinglePhi_ - nDoublePhi_ )/18 :
		 ( nSinglePhi_ - 1 < (int)din ?
		   firstHEDoublePhiRing_ + ( din - nSinglePhi_ )/36 :
		   din/72 + 1 ) ) ;
  
  const int ip ( nSinglePhi_ + nDoublePhi_ - 1 < (int)(din) ?
		 ( ( din - nSinglePhi_ - nDoublePhi_ )%18 )*4 + 3 :
		 ( nSinglePhi_ - 1 < (int)(din) ?
		   ( ( din - nSinglePhi_ )%36 )*2 + 1 :
		   din%72 + 1 ) ) ;

  return ( validDenseIndex( din ) ? CaloTowerDetId( iz*ie, ip ) : CaloTowerDetId() ) ;
}
