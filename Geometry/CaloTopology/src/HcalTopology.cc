#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include <cmath>
#include <iostream>
#include <cassert>
#include <algorithm>
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCalibDetId.h"
#include "Geometry/CaloTopology/interface/HcalTopologyRestrictionParser.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

static const int IPHI_MAX=72;
//#define DebugLog

HcalTopology::HcalTopology(const HcalDDDRecConstants* hcons, HcalTopologyMode::TriggerMode tmode) :
  hcons_(hcons),
  excludeHB_(false),  excludeHE_(false),  excludeHO_(false),  excludeHF_(false),
  triggerMode_(tmode),
  firstHBRing_(1), 
  firstHERing_(999), lastHERing_(0), 
  firstHFRing_(29), lastHFRing_(41),
  firstHORing_(1),  lastHORing_(15),
  firstHEDoublePhiRing_(999), firstHEQuadPhiRing_(999),
  firstHFQuadPhiRing_(40), firstHETripleDepthRing_(999),
  singlePhiBins_(72), doublePhiBins_(36) {

  StringToEnumParser<HcalTopologyMode::Mode> eparser;
  mode_       = eparser.parseString(hcons_->getTopoMode());
  maxDepthHB_ = hcons_->getMaxDepth(0);
  maxDepthHE_ = hcons_->getMaxDepth(1);
  etaBinsHB_  = hcons_->getEtaBins(0);
  etaBinsHE_  = hcons_->getEtaBins(1);
  nEtaHB_     = (int)(etaBinsHB_.size());
  lastHBRing_ = firstHBRing_+nEtaHB_-1;
  for (int i = 0; i < (int)(etaBinsHE_.size()); ++i) {
    if (firstHERing_ > etaBinsHE_[i].ieta) firstHERing_ = etaBinsHE_[i].ieta;
    if (lastHERing_  < etaBinsHE_[i].ieta) lastHERing_  = etaBinsHE_[i].ieta;
    int unit = (int)((etaBinsHE_[i].dphi+0.01)/(5.0*CLHEP::deg));
    if (unit == 2 && firstHEDoublePhiRing_ > etaBinsHE_[i].ieta)
      firstHEDoublePhiRing_ = etaBinsHE_[i].ieta;
    if (unit == 4 && firstHEQuadPhiRing_ > etaBinsHE_[i].ieta)
      firstHEQuadPhiRing_ = etaBinsHE_[i].ieta;
    if (etaBinsHE_[i].layer.size() > 2 && firstHETripleDepthRing_ > etaBinsHE_[i].ieta) 
      firstHETripleDepthRing_ = etaBinsHE_[i].ieta;
  }
  nEtaHE_     = (lastHERing_ - firstHERing_ + 1);
  if (mode_==HcalTopologyMode::LHC) {
    topoVersion_=0; //DL
    HBSize_     = kHBSizePreLS1; // qie-per-fiber * fiber/rm * rm/rbx * rbx/barrel * barrel/hcal
    HESize_     = kHESizePreLS1; // qie-per-fiber * fiber/rm * rm/rbx * rbx/endcap * endcap/hcal
    HOSize_     = kHOSizePreLS1; // ieta * iphi * 2
    HFSize_     = kHFSizePreLS1; // phi * eta * depth * pm 
    HTSize_     = kHTSizePreLS1; 
    numberOfShapes_ = 87;
  } else if (mode_==HcalTopologyMode::SLHC) { // need to know more eventually
    topoVersion_=10;
    HBSize_     = nEtaHB_*72*maxDepthHB_*2;
    HESize_     = nEtaHE_*72*maxDepthHE_*2;
    HOSize_     = (lastHORing_-firstHORing_+1)*72*2; // ieta * iphi * 2
    HFSize_     = (lastHFRing_-firstHFRing_+1)*72*2*2; // ieta * iphi * depth * 2
    HTSize_     = kHTSizePreLS1;  //no clue!
    numberOfShapes_ = 500;
  }
  maxEta_ = (lastHERing_ > lastHFRing_) ? lastHERing_ : lastHFRing_;

  //The transition between HE/HF in eta
  etaTableHF  = hcons_->getEtaTableHF();
  etaTable    = hcons_->getEtaTable();
  dPhiTableHF = hcons_->getPhiTableHF();
  dPhiTable   = hcons_->getPhiTable();
  phioff      = hcons_->getPhiOffs();
  std::pair<int,int>  ietaHF = hcons_->getEtaRange(2);
  double eta  = etaBinsHE_[etaBinsHE_.size()-1].etaMax;
  etaHE2HF_   = firstHFRing_;
  for (unsigned int i=1; i<etaTableHF.size(); ++i) {
    if (eta < etaTableHF[i]) {
      etaHE2HF_ = ietaHF.first + i - 1;
      break;
    }
  }
  eta         = etaTableHF[0];
  etaHF2HE_   = lastHERing_;
  for (unsigned int i=0; i<etaBinsHE_.size(); ++i) {
    if (eta < etaBinsHE_[i].etaMax) {
      etaHF2HE_ = etaBinsHE_[i].ieta;
      break;
    }
  }
  const double fiveDegInRad = 2*M_PI/72;
  for (unsigned int k=0; k<dPhiTable.size(); ++k) {
    int units = (int)(dPhiTable[k]/fiveDegInRad+0.5);
    unitPhi.push_back(units);
  }
  for (unsigned int k=0; k<dPhiTableHF.size(); ++k) {
    int units = (int)(dPhiTableHF[k]/fiveDegInRad+0.5);
    unitPhiHF.push_back(units);
  }
  int nEta = hcons_->getNEta();
  for (int ring=1; ring<=nEta; ++ring) {
    std::vector<int> segmentation = hcons_->getDepth(ring-1);
    setDepthSegmentation(ring,segmentation);
#ifdef DebugLog
    std::cout << "Set segmentation for ring " << ring << " with " 
	      << segmentation.size() << " elements:";
    for (unsigned int k=0; k<segmentation.size(); ++k) 
      std::cout << " " << segmentation[k];
    std::cout << std::endl;
#endif
  }
#ifdef DebugLog
  std::cout << "Constants in HcalTopology " << firstHBRing_ << ":" 
	    << lastHBRing_ << " " << firstHERing_ << ":" << lastHERing_ << ":" 
	    << firstHEDoublePhiRing_ << ":" << firstHEQuadPhiRing_ << ":" 
	    << firstHETripleDepthRing_ << " " << firstHFRing_ << ":" 
	    << lastHFRing_ << ":" << firstHFQuadPhiRing_ << " " << firstHORing_
	    << ":" << lastHORing_ << " " << maxDepthHB_ << ":" << maxDepthHE_ 
	    << " " << nEtaHB_ << ":" << nEtaHE_ << " " << etaHE2HF_ << ":" 
	    << etaHF2HE_ << std::endl;
#endif
}

HcalTopology::HcalTopology(HcalTopologyMode::Mode mode, int maxDepthHB, int maxDepthHE, HcalTopologyMode::TriggerMode tmode) :
  hcons_(0),
  excludeHB_(false),
  excludeHE_(false),
  excludeHO_(false),
  excludeHF_(false),
  mode_(mode),
  triggerMode_(tmode),
  firstHBRing_(1),   lastHBRing_(16),
  firstHERing_(16),  lastHERing_(29),
  firstHFRing_(29),  lastHFRing_(41),
  firstHORing_(1),   lastHORing_(15),
  firstHEDoublePhiRing_((mode==HcalTopologyMode::H2 || mode==HcalTopologyMode::H2HE)?(22):(21)),
  firstHEQuadPhiRing_(999), firstHFQuadPhiRing_(40),
  firstHETripleDepthRing_((mode==HcalTopologyMode::H2 || mode==HcalTopologyMode::H2HE)?(24):(27)),
  singlePhiBins_(72),
  doublePhiBins_(36),
  maxDepthHB_(maxDepthHB),
  maxDepthHE_(maxDepthHE),
  etaHE2HF_(30), etaHF2HE_(29),
  HBSize_(kHBSizePreLS1),
  HESize_(kHESizePreLS1),
  HOSize_(kHOSizePreLS1),
  HFSize_(kHFSizePreLS1),
  HTSize_(kHTSizePreLS1),
  numberOfShapes_(( mode==HcalTopologyMode::SLHC ) ? 500 : 87 ) {

  if (mode_==HcalTopologyMode::LHC) {
    topoVersion_=0; //DL
    HBSize_= kHBSizePreLS1; // qie-per-fiber * fiber/rm * rm/rbx * rbx/barrel * barrel/hcal
    HESize_= kHESizePreLS1; // qie-per-fiber * fiber/rm * rm/rbx * rbx/endcap * endcap/hcal
    HOSize_= kHOSizePreLS1; // ieta * iphi * 2
    HFSize_= kHFSizePreLS1; // phi * eta * depth * pm 
    HTSize_= kHTSizePreLS1; 
  } else if (mode_==HcalTopologyMode::SLHC) { // need to know more eventually
    HBSize_= maxDepthHB*16*72*2;
    HESize_= maxDepthHE*(29-16+1)*72*2;
    HOSize_= 15*72*2; // ieta * iphi * 2
    HFSize_= 72*13*2*2; // phi * eta * depth * pm 
    HTSize_= kHTSizePreLS1;  //no clue!

    topoVersion_=10;
  }
  nEtaHB_ = (lastHBRing_-firstHBRing_+1);
  nEtaHE_ = (lastHERing_-firstHERing_+1);

  edm::LogWarning("CaloTopology") << "This is an incomplete constructor of HcalTopology - be warned that many functionalities will not be there - revert from this - get from EventSetup";

}

bool HcalTopology::valid(const DetId& id) const {
  assert(id.det()==DetId::Hcal);
  return validHcal(id);
}

bool HcalTopology::validHcal(const HcalDetId& id) const {
  // check the raw rules
  bool ok=validRaw(id);

  ok=ok && !isExcluded(id);

  return ok;
}

bool HcalTopology::isExcluded(const HcalDetId& id) const {
  bool exed=false;
  // first, check the full detector exclusions...  (fast)
  switch (id.subdet()) {
  case(HcalBarrel):  exed=excludeHB_; break;
  case(HcalEndcap):  exed=excludeHE_; break;
  case(HcalOuter):   exed=excludeHO_; break;
  case(HcalForward): exed=excludeHF_; break;
  default: exed=false;
  }
  // next, check the list (slower)
  if (!exed && !exclusionList_.empty()) {
    std::vector<HcalDetId>::const_iterator i=std::lower_bound(exclusionList_.begin(),exclusionList_.end(),id);
    if (i!=exclusionList_.end() && *i==id) exed=true;
  }
  return exed;
}

void HcalTopology::exclude(const HcalDetId& id) {
  std::vector<HcalDetId>::iterator i=std::lower_bound(exclusionList_.begin(),exclusionList_.end(),id);
  if (i==exclusionList_.end() || *i!=id) {
    exclusionList_.insert(i,id);
  }
}

void HcalTopology::excludeSubdetector(HcalSubdetector subdet) {
  switch (subdet) {
  case(HcalBarrel):  excludeHB_=true; break;
  case(HcalEndcap):  excludeHE_=true; break;
  case(HcalOuter):   excludeHO_=true; break;
  case(HcalForward): excludeHF_=true; break;
  default: break;
  }
}

std::vector<DetId> HcalTopology::east(const DetId& id) const {
  std::vector<DetId> vNeighborsDetId;
  HcalDetId neighbors[2];
  for (int i=0;i<decIEta(HcalDetId(id),neighbors);i++)
    vNeighborsDetId.push_back(DetId(neighbors[i].rawId()));
  return vNeighborsDetId;
}

std::vector<DetId> HcalTopology::west(const DetId& id) const {
  std::vector<DetId> vNeighborsDetId;
  HcalDetId neighbors[2];
  for (int i=0;i<incIEta(HcalDetId(id),neighbors);i++)
    vNeighborsDetId.push_back(DetId(neighbors[i].rawId()));
  return  vNeighborsDetId;
}

std::vector<DetId> HcalTopology::north(const DetId& id) const {
  std::vector<DetId> vNeighborsDetId;
  HcalDetId neighbor;
  if (incIPhi(HcalDetId(id),neighbor))
    vNeighborsDetId.push_back(DetId(neighbor.rawId()));
  return  vNeighborsDetId;
}

std::vector<DetId> HcalTopology::south(const DetId& id) const {
  std::vector<DetId> vNeighborsDetId;
  HcalDetId neighbor;
  if (decIPhi(HcalDetId(id),neighbor))
    vNeighborsDetId.push_back(DetId(neighbor.rawId()));
  return  vNeighborsDetId;
}

std::vector<DetId> HcalTopology::up(const DetId& id) const {
  HcalDetId neighbor = id;
  std::vector<DetId> vNeighborsDetId;
  if (incrementDepth(neighbor)) {
    vNeighborsDetId.push_back(neighbor);
  }
  return  vNeighborsDetId;
}

std::vector<DetId> HcalTopology::down(const DetId& id) const {
  HcalDetId neighbor = id;
  std::vector<DetId> vNeighborsDetId;
  if (decrementDepth(neighbor)) {
    vNeighborsDetId.push_back(neighbor);
  }
  return  vNeighborsDetId;
}

int HcalTopology::exclude(HcalSubdetector subdet, int ieta1, int ieta2, int iphi1, int iphi2, int depth1, int depth2) {

  bool exed=false;
  // first, check the full detector exclusions...  (fast)
  switch (subdet) {
  case(HcalBarrel):  exed=excludeHB_; break;
  case(HcalEndcap):  exed=excludeHE_; break;
  case(HcalOuter):   exed=excludeHO_; break;
  case(HcalForward): exed=excludeHF_; break;
  default: exed=false;
  }
  if (exed) return 0; // if the whole detector is excluded...

  int ieta_l=std::min(ieta1,ieta2);
  int ieta_h=std::max(ieta1,ieta2);
  int iphi_l=std::min(iphi1,iphi2);
  int iphi_h=std::max(iphi1,iphi2);
  int depth_l=std::min(depth1,depth2);
  int depth_h=std::max(depth1,depth2);

  int n=0;
  for (int ieta=ieta_l; ieta<=ieta_h; ieta++) 
    for (int iphi=iphi_l; iphi<=iphi_h; iphi++) 
      for (int depth=depth_l; depth<=depth_h; depth++) {
	HcalDetId id(subdet,ieta,iphi,depth);
	if (validRaw(id)) { // use 'validRaw' to include check validity in "uncut" detector
	  exclude(id);  
	  n++;
	}
      }
  return n;
}

  /** Basic rules used to derive this code:
      
  HB has 72 towers in iphi.  Ieta 1-14 have depth=1, Ieta 15-16 have depth=1 or 2.

  HE ieta=16-20 have 72 towers in iphi
     ieta=21-29 have 36 towers in iphi
     ieta=16 is depth 3 only
     ieta=17 is depth 1 only
     ieta=18-26 & 29 have depth 1 and 2
     ieta=27-28 has depth 1-3

  HF ieta=29-39 have 36 in iphi
     ieta=40-41 have 18 in iphi (71,3,7,11...)
     all have two depths


  HO has 15 towers in ieta and 72 in iphi and depth = 4 (one value)

  At H2:

  HE ieta 17 is two depths
  HE ieta 22- have 36 towers in iphi (starts one higher)
  HE ieta 24- has three depths

  */

bool HcalTopology::validDetIdPreLS1(const HcalDetId& id) const {
  const HcalSubdetector sd (id.subdet());
  const int             ie (id.ietaAbs());
  const int             ip (id.iphi());
  const int             dp (id.depth());

  return ( ( ip >=  1         ) &&
	   ( ip <= IPHI_MAX   ) &&
	   ( dp >=  1         ) &&
	   ( ie >=  1         ) &&
	   ( ( ( sd == HcalBarrel ) &&
	       ( ( ( ie <= 14         ) &&
		   ( dp ==  1         )    ) ||
		 ( ( ( ie == 15 ) || ( ie == 16 ) ) && 
		   ( dp <= 2          )                ) ) ) ||
	     (  ( sd == HcalEndcap ) &&
		( ( ( ie == firstHERing() ) &&
		    ( dp ==  3 )          ) ||
		  ( ( ie == 17 ) &&
		    ( dp ==  1 )          ) ||
		  ( ( ie >= 18 ) &&
		    ( ie <= 20 ) &&
		    ( dp <=  2 )          ) ||
		  ( ( ie >= 21 ) &&
		    ( ie <= 26 ) &&
		    ( dp <=  2 ) &&
		    ( ip%2 == 1 )         ) ||
		  ( ( ie >= 27 ) &&
		    ( ie <= 28 ) &&
		    ( dp <=  3 ) &&
		    ( ip%2 == 1 )         ) ||
		  ( ( ie == 29 ) &&
		    ( dp <=  2 ) &&
		    ( ip%2 == 1 )         )          )      ) ||
	     (  ( sd == HcalOuter ) &&
		( ie <= 15 ) &&
		( dp ==  4 )           ) ||
	     (  ( sd == HcalForward ) &&
		( dp <=  2 )          &&
		( ( ( ie >= firstHFRing() ) &&
		    ( ie <  firstHFQuadPhiRing() ) &&
		    ( ip%2 == 1 )    ) ||
		  ( ( ie >= firstHFQuadPhiRing() ) &&
		    ( ie <= lastHFRing() ) &&
		    ( ip%4 == 3 )         )  ) ) ) ) ;
}

  /** Is this a valid cell id? */
bool HcalTopology::validRaw(const HcalDetId& id) const {
  bool ok=true;
  int ieta=id.ieta();
  int aieta=id.ietaAbs();
  int depth=id.depth();
  int iphi=id.iphi();
  if ((ieta==0 || iphi<=0 || iphi>IPHI_MAX) || aieta>maxEta_) return false; // outer limits
 
  if (ok) {
    HcalSubdetector subdet=id.subdet();
    if (subdet==HcalBarrel) {
      if (mode_==HcalTopologyMode::SLHC || mode_==HcalTopologyMode::H2HE) {
	if ((aieta>lastHBRing() || depth>maxDepthHB_ || (aieta==lastHBRing() && depth > 2))) ok=false;
      } else {
	if (aieta>lastHBRing() || depth>2 || (aieta<=14 && depth>1)) ok=false;
      }
    } else if (subdet==HcalEndcap) {
      if (mode_==HcalTopologyMode::SLHC || mode_==HcalTopologyMode::H2HE) {
	if (depth>maxDepthHE_ || aieta<firstHERing() || aieta>lastHERing() ||
	    (aieta==firstHERing() && depth<3)) {
	  ok = false;
	} else {
	  for (unsigned int i=0; i<etaBinsHE_.size(); ++i) {
	    if (aieta == etaBinsHE_[i].ieta) {
	      if (aieta >= firstHEDoublePhiRing() && (iphi%2)==0) ok=false;
	      if (aieta >= firstHEQuadPhiRing()   && (iphi%4)!=3) ok=false;
	      if (depth < etaBinsHE_[i].depthStart || 
		  depth > (etaBinsHE_[i].depthStart+(int)(etaBinsHE_[i].layer.size())))
		ok = false;
	      break;
	    }
	  }
	}
      } else {
	if (depth>3 || aieta<firstHERing() || aieta>lastHERing() || (aieta==firstHERing() && depth!=3) || (aieta==17 && depth!=1 && mode_!=HcalTopologyMode::H2) || // special case at H2
	    (((aieta>=17 && aieta<firstHETripleDepthRing()) || aieta==lastHERing()) && depth>2) ||
	    (aieta>=firstHEDoublePhiRing() && (iphi%2)==0)) ok=false;
      }
    } else if (subdet==HcalOuter) {
      if (aieta>lastHORing() || iphi>IPHI_MAX || depth!=4) ok=false;
    } else if (subdet==HcalForward) {
      if (aieta<firstHFRing() || aieta>lastHFRing() || ((iphi%2)==0) || (depth>2) ||  (aieta>=firstHFQuadPhiRing() && ((iphi+1)%4)!=0)) ok=false;
    } else {
      ok=false;
    }
  }
  return ok;
}

bool HcalTopology::incIPhi(const HcalDetId& id, HcalDetId &neighbor) const {
  bool ok=valid(id);
  if (ok) {
    switch (id.subdet()) {
    case (HcalBarrel):
    case (HcalOuter):
      if (id.iphi()==IPHI_MAX) neighbor=HcalDetId(id.subdet(),id.ieta(),1,id.depth()); 
      else                     neighbor=HcalDetId(id.subdet(),id.ieta(),id.iphi()+1,id.depth()); 
      break;
    case (HcalEndcap):
      if (id.ietaAbs()>=firstHEQuadPhiRing()) {
	if (id.iphi()==IPHI_MAX-1) neighbor=HcalDetId(id.subdet(),id.ieta(),3,id.depth()); 
	else                       neighbor=HcalDetId(id.subdet(),id.ieta(),id.iphi()+4,id.depth()); 
      } else if (id.ietaAbs()>=firstHEDoublePhiRing()) {
	if (id.iphi()==IPHI_MAX-1) neighbor=HcalDetId(id.subdet(),id.ieta(),1,id.depth()); 
	else                       neighbor=HcalDetId(id.subdet(),id.ieta(),id.iphi()+2,id.depth()); 
      } else {
	if (id.iphi()==IPHI_MAX) neighbor=HcalDetId(id.subdet(),id.ieta(),1,id.depth()); 
	else                     neighbor=HcalDetId(id.subdet(),id.ieta(),id.iphi()+1,id.depth()); 
      }	
      break;
    case (HcalForward):
      if (id.ietaAbs()>=firstHFQuadPhiRing()) {
	if (id.iphi()==IPHI_MAX-1) neighbor=HcalDetId(id.subdet(),id.ieta(),3,id.depth()); 
	else                       neighbor=HcalDetId(id.subdet(),id.ieta(),id.iphi()+4,id.depth()); 
      } else {
	if (id.iphi()==IPHI_MAX-1) neighbor=HcalDetId(id.subdet(),id.ieta(),1,id.depth()); 
	else                       neighbor=HcalDetId(id.subdet(),id.ieta(),id.iphi()+2,id.depth()); 
      }
      break;
    default: ok=false;
    }
  } 
  return ok;
}

/** Get the neighbor (if present) of the given cell with lower iphi */
bool HcalTopology::decIPhi(const HcalDetId& id, HcalDetId &neighbor) const {
  bool ok=valid(id);
  if (ok) {
    switch (id.subdet()) {
    case (HcalBarrel):
    case (HcalOuter):
      if (id.iphi()==1) neighbor=HcalDetId(id.subdet(),id.ieta(),IPHI_MAX,id.depth()); 
      else              neighbor=HcalDetId(id.subdet(),id.ieta(),id.iphi()-1,id.depth()); 
      break;
    case (HcalEndcap):
      if (id.ietaAbs()>=firstHEQuadPhiRing()) {
	if (id.iphi()==3) neighbor=HcalDetId(id.subdet(),id.ieta(),IPHI_MAX-1,id.depth()); 
	else              neighbor=HcalDetId(id.subdet(),id.ieta(),id.iphi()-4,id.depth()); 
      } else if (id.ietaAbs()>=firstHEDoublePhiRing()) {
	if (id.iphi()==1) neighbor=HcalDetId(id.subdet(),id.ieta(),IPHI_MAX-1,id.depth()); 
	else              neighbor=HcalDetId(id.subdet(),id.ieta(),id.iphi()-2,id.depth()); 
      } else {
	if (id.iphi()==1) neighbor=HcalDetId(id.subdet(),id.ieta(),IPHI_MAX,id.depth()); 
	else              neighbor=HcalDetId(id.subdet(),id.ieta(),id.iphi()-1,id.depth()); 
      }
      break;
    case (HcalForward):
      if (id.ietaAbs()>=firstHFQuadPhiRing()) {
	if (id.iphi()==3) neighbor=HcalDetId(id.subdet(),id.ieta(),IPHI_MAX-1,id.depth()); 
	else              neighbor=HcalDetId(id.subdet(),id.ieta(),id.iphi()-4,id.depth()); 
      } else {
	if (id.iphi()==1) neighbor=HcalDetId(id.subdet(),id.ieta(),IPHI_MAX-1,id.depth()); 
	else              neighbor=HcalDetId(id.subdet(),id.ieta(),id.iphi()-2,id.depth()); 
      }
      break;
    default: ok=false;
    }
  } 
  return ok;
}

int HcalTopology::incIEta(const HcalDetId& id, HcalDetId neighbors[2]) const {
  if (id.zside()==1) return incAIEta(id,neighbors);
  else               return decAIEta(id,neighbors);
}

int HcalTopology::decIEta(const HcalDetId& id, HcalDetId neighbors[2]) const {
  if (id.zside()==1) return decAIEta(id,neighbors);
  else               return incAIEta(id,neighbors);
}

/** Increasing in |ieta|, there is always at most one neighbor */
int HcalTopology::incAIEta(const HcalDetId& id, HcalDetId neighbors[2]) const {
  int n=1;
  int aieta=id.ietaAbs();

  if (aieta==firstHEDoublePhiRing()-1 && (id.iphi()%2)==0) 
    neighbors[0]=HcalDetId(id.subdet(),(aieta+1)*id.zside(),id.iphi()-1,id.depth());
  else if (aieta==firstHFQuadPhiRing()-1 && ((id.iphi()+1)%4)!=0) 
    neighbors[0]=HcalDetId(id.subdet(),(aieta+1)*id.zside(),((id.iphi()==1)?(71):(id.iphi()-2)),id.depth());
  else if (aieta==firstHEQuadPhiRing()-1 && ((id.iphi()+1)%4)!=0) 
    neighbors[0]=HcalDetId(id.subdet(),(aieta+1)*id.zside(),((id.iphi()==1)?(71):(id.iphi()-2)),id.depth());
  else if (aieta==lastHBRing()) 
    neighbors[0]=HcalDetId(HcalEndcap,(aieta+1)*id.zside(),id.iphi(),1);
  else if (aieta==lastHERing()) 
    neighbors[0]=HcalDetId(HcalForward,etaHE2HF_*id.zside(),id.iphi(),1);
  else
    neighbors[0]=HcalDetId(id.subdet(),(aieta+1)*id.zside(),id.iphi(),id.depth());
    
  if (!valid(neighbors[0])) n=0;
  return n;
}

/** Decreasing in |ieta|, there are be two neighbors of 40 and 21*/
int HcalTopology::decAIEta(const HcalDetId& id, HcalDetId neighbors[2]) const {
  int n=1;
  int aieta=id.ietaAbs();

  if (aieta==firstHEDoublePhiRing()) { 
    n=2;
    neighbors[0]=HcalDetId(id.subdet(),(aieta-1)*id.zside(),id.iphi(),id.depth());
    neighbors[1]=HcalDetId(id.subdet(),(aieta-1)*id.zside(),id.iphi()+1,id.depth());
  } else if (aieta==firstHFQuadPhiRing()) {
    n=2;
    neighbors[0]=HcalDetId(id.subdet(),(aieta-1)*id.zside(),id.iphi(),id.depth());
    if (id.iphi()==IPHI_MAX-1) neighbors[1]=HcalDetId(id.subdet(),(aieta-1)*id.zside(),1,id.depth());
    else neighbors[1]=HcalDetId(id.subdet(),(aieta-1)*id.zside(),id.iphi()+2,id.depth());
  } else if (aieta==firstHEQuadPhiRing()) {
    n=2;
    neighbors[0]=HcalDetId(id.subdet(),(aieta-1)*id.zside(),id.iphi(),id.depth());
    if (id.iphi()==IPHI_MAX-1) neighbors[1]=HcalDetId(id.subdet(),(aieta-1)*id.zside(),1,id.depth());
    else                       neighbors[1]=HcalDetId(id.subdet(),(aieta-1)*id.zside(),id.iphi()+2,id.depth());
  } else if (aieta==1) {
    neighbors[0]=HcalDetId(id.subdet(),-aieta*id.zside(),id.iphi(),id.depth());
  } else if (aieta==firstHERing()) {
    neighbors[0]=HcalDetId(HcalBarrel,(aieta-1)*id.zside(),id.iphi(),1);
  } else if (aieta==firstHFRing()) {
    neighbors[0]=HcalDetId(HcalEndcap,etaHF2HE_*id.zside(),id.iphi(),1);
  } else
    neighbors[0]=HcalDetId(id.subdet(),(aieta-1)*id.zside(),id.iphi(),id.depth());
  
  if (!valid(neighbors[0]) && n==2) {
    if (!valid(neighbors[1])) n=0;
    else {
      n=1;
      neighbors[0]=neighbors[1];
    }
  }
  if (n==2 && !valid(neighbors[1])) n=1;
  if (n==1 && !valid(neighbors[0])) n=0;

  return n;
}


void HcalTopology::depthBinInformation(HcalSubdetector subdet, int etaRing,
                                       int & nDepthBins, int & startingBin) const {

  if(subdet == HcalBarrel) {
    if (mode_==HcalTopologyMode::SLHC || mode_==HcalTopologyMode::H2HE) {
      startingBin = 1;
      if (etaRing==lastHBRing()) {
	nDepthBins = 2;
      } else {
	nDepthBins = maxDepthHB_;
      }
    } else {
      if (etaRing<=14) {
	nDepthBins  = 1;
	startingBin = 1;
      } else {
	nDepthBins  = 2;
	startingBin = 1;
      }
    }
  } else if(subdet == HcalEndcap) {
    if (mode_==HcalTopologyMode::SLHC || mode_==HcalTopologyMode::H2HE) {
      if (etaRing==firstHERing()) {
	nDepthBins  = 1;
	startingBin = 3;
      } else {
	nDepthBins  = maxDepthHE_;
	startingBin = 1;
      }
    } else {
      if (etaRing==firstHERing()) {
	nDepthBins  = 1;
	startingBin = 3;
      } else if (etaRing==17) {
	nDepthBins  = 1;
	startingBin = 1;
      } else if (etaRing==lastHERing()) {
	nDepthBins  = 2;
	startingBin = 1;
      }	else {
	nDepthBins  = (etaRing >= firstHETripleDepthRing()) ? 3 : 2;
	startingBin = 1;
      }
    }
  } else if(subdet == HcalForward) {
    nDepthBins  = 2;
    startingBin = 1;
  } else if(subdet == HcalOuter) {
    nDepthBins  = 1;
    startingBin = 4;
  } else {
    std::cerr << "Bad HCAL subdetector " << subdet << std::endl;
  }
}


bool HcalTopology::incrementDepth(HcalDetId & detId) const {
  HcalSubdetector subdet = detId.subdet();
  int ieta    = detId.ieta();
  int etaRing = detId.ietaAbs();
  int depth   = detId.depth();
  int nDepthBins, startingBin;
  depthBinInformation(subdet, etaRing, nDepthBins, startingBin);

  // see if the new depth bin exists
  ++depth;
  if (depth > nDepthBins) {
    // handle on a case-by-case basis
    if (subdet == HcalBarrel && etaRing < lastHORing())  {
      // HO
      subdet = HcalOuter;
      depth  = 4;
    } else if (subdet == HcalBarrel && etaRing == lastHBRing()) {
      // overlap
      subdet = HcalEndcap;
    } else if (subdet == HcalEndcap && etaRing ==  lastHERing()-1 &&
	       mode_ != HcalTopologyMode::SLHC) {
      // guard ring HF29 is behind HE 28
      subdet = HcalForward;
      (ieta > 0) ? ++ieta : --ieta;
      depth = 1;
    } else if (subdet == HcalEndcap && etaRing ==  lastHERing() &&
	       mode_ != HcalTopologyMode::SLHC) {
      // split cells go to bigger granularity.  Ring 29 -> 28
      (ieta > 0) ? --ieta : ++ieta;
    } else {
      // no more chances
      detId = HcalDetId();
      return false;
    }
  }
  detId = HcalDetId(subdet, ieta, detId.iphi(), depth);
  return validRaw(detId);
}

bool HcalTopology::decrementDepth(HcalDetId & detId) const {
  HcalSubdetector subdet = detId.subdet();
  int ieta    = detId.ieta();
  int etaRing = detId.ietaAbs();
  int depth   = detId.depth();
  int nDepthBins, startingBin;
  depthBinInformation(subdet, etaRing, nDepthBins, startingBin);

  // see if the new depth bin exists
  --depth;
  if ((subdet == HcalOuter) ||
      (subdet == HcalEndcap && etaRing == firstHERing())) {
    subdet = HcalBarrel;
    for (int i=0; i<nEtaHB_; ++i) {
      if (etaRing == etaBinsHB_[i].ieta) {
	depth = etaBinsHB_[i].depthStart+etaBinsHB_[i].layer.size()-1;
	break;
      }
    }
  } else if (subdet == HcalEndcap && etaRing ==  lastHERing() && depth == 2 &&
	     mode_ != HcalTopologyMode::SLHC) {
    (ieta > 0) ? --ieta : ++ieta;
  } else if (depth <= 0) {
    if (subdet == HcalForward && etaRing ==  firstHFRing()) {
      // overlap
      subdet = HcalEndcap;
      etaRing= etaHF2HE_;
      ieta   = (ieta > 0) ? etaRing : -etaRing;
      for (unsigned int i=0; i<etaBinsHE_.size(); ++i) {
	if (etaRing == etaBinsHE_[i].ieta) {
	  depth = etaBinsHE_[i].depthStart+etaBinsHE_[i].layer.size()-1;
	  break;
	}
      }
    } else {
      // no more chances
      detId = HcalDetId();
      return false;
    }
  }
  detId = HcalDetId(subdet, ieta, detId.iphi(), depth);
  return validRaw(detId);
}

int HcalTopology::nPhiBins(int etaRing) const {
  int lastPhiBin=singlePhiBins_;
  if      (etaRing>= firstHFQuadPhiRing())   lastPhiBin=doublePhiBins_/2;
  else if (etaRing>= firstHEDoublePhiRing()) lastPhiBin=doublePhiBins_;
  return lastPhiBin;
}

int HcalTopology::nPhiBins(HcalSubdetector bc, int etaRing) const {
  static const double twopi = M_PI+M_PI;
  int lastPhiBin=singlePhiBins_;
  if (bc == HcalForward) {
    lastPhiBin = (int)((twopi+0.001)/dPhiTableHF[etaRing-firstHFRing_]);
  } else {
    lastPhiBin = (int)((twopi+0.001)/dPhiTable[etaRing-firstHBRing_]);
  }
  return lastPhiBin;
}

int HcalTopology::etaRing(HcalSubdetector bc, double abseta) const {

  int etaring = firstHBRing_;
  if (bc == HcalForward) {
    etaring = firstHFRing_;
    for (unsigned int k=0; k<etaTableHF.size()-1; ++k) {
      if (abseta < etaTableHF[k+1]) {
	etaring += k;
        break;
      }
    }
  } else {
    for (unsigned int k=0; k<etaTable.size()-1; ++k) {
      if (abseta < etaTable[k+1]) {
	etaring += k;
        break;
      }
    }
  }
  return etaring;
}

int HcalTopology::phiBin(HcalSubdetector bc, int etaring, double phi) const {
  static const double twopi = M_PI+M_PI;
  //put phi in correct range (0->2pi)
  int index(0);
  if (bc == HcalBarrel) {
    index = (etaring-firstHBRing_);
    phi  -= phioff[0];
  } else if (bc == HcalEndcap) {
    index = (etaring-firstHBRing_);
    phi  -= phioff[1];
  } else if (bc == HcalForward) {
    index = (etaring-firstHFRing_);
    if (index < (int)(dPhiTableHF.size())) {
      if (unitPhiHF[index] > 2) phi -= phioff[4];
      else                      phi -= phioff[2];
    }
  }
  if (phi<0.0)   phi += twopi;
  if (phi>twopi) phi -= twopi;
  int phibin(1), unit(1);
  if (bc == HcalForward) {
    if (index < (int)(dPhiTableHF.size())) {
      unit    = unitPhiHF[index];
      phibin  = static_cast<int>(phi/dPhiTableHF[index])+1;
    }
  } else {
    if (index < (int)(dPhiTable.size())) {
      phibin  = static_cast<int>(phi/dPhiTable[index])+1;
      unit    = unitPhi[index];
    }
  }
  int iphi(phibin);
  if      (unit == 2) iphi = (phibin-1)*2+1;
  else if (unit == 4) iphi = (phibin-1)*4+3;
  return iphi;
}

void HcalTopology::getDepthSegmentation(unsigned ring, std::vector<int> & readoutDepths) const {
  // if it doesn't exist, return the first entry with a lower index.  So if we only
  // have entries for 1 and 17, any input from 1-16 should return the entry for ring 1
  SegmentationMap::const_iterator pos = depthSegmentation_.upper_bound(ring);
  if (pos == depthSegmentation_.begin()) {
    throw cms::Exception("HcalTopology") << "No depth segmentation found for ring" << ring;
  }
  --pos;
    // pos now refers to the last element with key <= ring.
  readoutDepths = pos->second;
}

void HcalTopology::setDepthSegmentation(unsigned ring, const std::vector<int> & readoutDepths) {
  depthSegmentation_[ring] = readoutDepths;
}

std::pair<int, int> HcalTopology::segmentBoundaries(unsigned ring, unsigned depth) const {
  std::vector<int> readoutDepths;
  getDepthSegmentation(ring, readoutDepths);
  int d1 = std::lower_bound(readoutDepths.begin(), readoutDepths.end(), depth) - readoutDepths.begin();
  int d2 = std::upper_bound(readoutDepths.begin(), readoutDepths.end(), depth) - readoutDepths.begin();
  return std::pair<int, int>(d1, d2);
}

unsigned int HcalTopology::detId2denseIdPreLS1 (const DetId& id) const {

  HcalDetId hid(id);
  const HcalSubdetector sd (hid.subdet()  ) ;
  const int             ip (hid.iphi()    ) ;
  const int             ie (hid.ietaAbs() ) ;
  const int             dp (hid.depth()   ) ;
  const int             zn (hid.zside() < 0 ? 1 : 0 ) ;
  unsigned int  retval = ( ( sd == HcalBarrel ) ?
			   ( ip - 1 )*18 + dp - 1 + ie - ( ie<16 ? 1 : 0 ) + zn*kHBhalf :
			   ( ( sd == HcalEndcap ) ?
			     2*kHBhalf + ( ip - 1 )*8 + ( ip/2 )*20 +
			     ( ( ie==16 || ie==17 ) ? ie - 16 :
			       ( ( ie>=18 && ie<=20 ) ? 2 + 2*( ie - 18 ) + dp - 1 :
				 ( ( ie>=21 && ie<=26 ) ? 8 + 2*( ie - 21 ) + dp - 1 :
				   ( ( ie>=27 && ie<=28 ) ? 20 + 3*( ie - 27 ) + dp - 1 :
				     26 + 2*( ie - 29 ) + dp - 1 ) ) ) ) + zn*kHEhalf :
			     ( ( sd == HcalOuter ) ?
			       2*kHBhalf + 2*kHEhalf + ( ip - 1 )*15 + ( ie - 1 ) + zn*kHOhalf :
			       ( ( sd == HcalForward ) ?
				 2*kHBhalf + 2*kHEhalf + 2*kHOhalf + 
				 ( ( ip - 1 )/4 )*4 + ( ( ip - 1 )/2 )*22 + 
				 2*( ie - 29 ) + ( dp - 1 ) + zn*kHFhalf : 0xFFFFFFFFu ) ) ) ) ; 
  return retval;
}


unsigned int HcalTopology::detId2denseIdHB(const DetId& id) const {
  HcalDetId hid(id);
  const int             ip (hid.iphi()    ) ;
  const int             ie (hid.ietaAbs() ) ;
  const int             dp (hid.depth()   ) ;
  const int             zn (hid.zside() < 0 ? 1 : 0 ) ;
  unsigned int  retval = 0xFFFFFFFFu;
  if (topoVersion_==0) {
    retval=( ip - 1 )*18 + dp - 1 + ie - ( ie<16 ? 1 : 0 ) + zn*kHBhalf;
  } else if (topoVersion_==10) {
    retval=(dp-1)+maxDepthHB_*(ip-1);
    if (hid.ieta()>0) retval+=maxDepthHB_*72*(hid.ieta()-firstHBRing());
    else              retval+=maxDepthHB_*72*(hid.ieta()+lastHBRing()+nEtaHB_);
  }
  return retval;
}

unsigned int HcalTopology::detId2denseIdHE(const DetId& id) const {
  HcalDetId hid(id);
  const int             ip (hid.iphi()    ) ;
  const int             ie (hid.ietaAbs() ) ;
  const int             dp (hid.depth()   ) ;
  const int             zn (hid.zside() < 0 ? 1 : 0 ) ;
  unsigned int  retval =  0xFFFFFFFFu;
  if (topoVersion_==0) {
    retval=( ip - 1 )*8 + ( ip/2 )*20 +
      ( ( ie==16 || ie==17 ) ? ie - 16 :
	( ( ie>=18 && ie<=20 ) ? 2 + 2*( ie - 18 ) + dp - 1 :
	  ( ( ie>=21 && ie<=26 ) ? 8 + 2*( ie - 21 ) + dp - 1 :
	    ( ( ie>=27 && ie<=28 ) ? 20 + 3*( ie - 27 ) + dp - 1 :
	      26 + 2*( ie - 29 ) + dp - 1 ) ) ) ) + zn*kHEhalf;
  } else if (topoVersion_==10) {
    retval=(dp-1)+maxDepthHE_*(ip-1);
    if (hid.ieta()>0) retval+=maxDepthHE_*72*(hid.ieta()-firstHERing());
    else              retval+=maxDepthHE_*72*(hid.ieta()+lastHERing()+nEtaHE_);
  }
  return retval;
}

unsigned int HcalTopology::detId2denseIdHO(const DetId& id) const {
  HcalDetId hid(id);
  const int             ip (hid.iphi()    ) ;
  const int             ie (hid.ietaAbs() ) ;
  const int             zn (hid.zside() < 0 ? 1 : 0 ) ;

  unsigned int  retval = 0xFFFFFFFFu;
  if (topoVersion_==0) {
    retval=( ip - 1 )*15 + ( ie - 1 ) + zn*kHOhalf;
  } else if (topoVersion_==10) {
    if   (hid.ieta()>0) retval=(ip-1)+72*(hid.ieta()-1);
    else                retval=(ip-1)+72*(30+hid.ieta());
  }
  return retval;
}

unsigned int HcalTopology::detId2denseIdHF(const DetId& id) const {
  HcalDetId hid(id);
  const int             ip (hid.iphi()    ) ;
  const int             ie (hid.ietaAbs() ) ;
  const int             dp (hid.depth()   ) ;
  const int             zn (hid.zside() < 0 ? 1 : 0 ) ;

  unsigned int  retval = 0xFFFFFFFFu;
  if (topoVersion_==0) {
    retval = ( ( ip - 1 )/4 )*4 + ( ( ip - 1 )/2 )*22 + 
      2*( ie - 29 ) + ( dp - 1 ) + zn*kHFhalf;
  } else if (topoVersion_==10) {
    retval=dp-1+2*(ip-1);
    if (hid.ieta()>0) retval+=2*72*(hid.ieta()-29);
    else              retval+=2*72*((41+13)+hid.ieta());
  }
  return retval;
}

unsigned int HcalTopology::detId2denseIdHT(const DetId& id) const {
  HcalTrigTowerDetId tid(id); 
  int zside = tid.zside();
  unsigned int ietaAbs = tid.ietaAbs();
  unsigned int iphi = tid.iphi();

  unsigned int index;
  if ((iphi-1)%4==0) index = (iphi-1)*32 + (ietaAbs-1) - (12*((iphi-1)/4));
  else               index = (iphi-1)*28 + (ietaAbs-1) + (4*(((iphi-1)/4)+1));
  
  if (zside == -1) index += kHThalf;

  return index;
}

unsigned int HcalTopology::detId2denseIdCALIB(const DetId& id) const {
  HcalCalibDetId tid(id);
  int    channel = tid.cboxChannel();
  int ieta = tid.ieta();
  int iphi = tid.iphi();
  int zside = tid.zside();
  unsigned int index=0xFFFFFFFFu;
      
  if (tid.calibFlavor()==HcalCalibDetId::CalibrationBox) {
        
    HcalSubdetector subDet = tid.hcalSubdet();
        
    if (subDet==HcalBarrel) {
      //std::cout<<"CALIB_HB:  ";
      //dphi = 4 (18 phi values), 3 channel types (0,1,2), eta = -1 or 1
      //total of 18*3*2=108 channels
      index = ((iphi+1)/4-1) + 18*channel + 27*(ieta+1);
    } else if (subDet==HcalEndcap) {
      //std::cout<<"CALIB_HE:  ";
      //dphi = 4 (18 phi values), 6 channel types (0,1,3,4,5,6), eta = -1 or 1
      //total of 18*6*2=216 channels
      if (channel>2) channel-=1;
      index = ((iphi+1)/4-1) + 18*channel + 54*(ieta+1) + 108;
    } else if (subDet==HcalForward) {
      //std::cout<<"CALIB_HF:  ";
      //dphi = 18 (4 phi values), 3 channel types (0,1,8), eta = -1 or 1
      if (channel==8) channel = 2;
      //total channels 4*3*2=24
      index = (iphi-1)/18 + 4*channel + 6*(ieta+1) + 324;
    } else if (subDet==HcalOuter) {
      //std::cout<<"CALIB_HO:  ";
      //there are 5 special calib crosstalk channels, one in each ring
      if (channel==7) {
	channel = 2;
	index = (ieta+2) + 420;
      //for HOM/HOP dphi = 6 (12 phi values),  2 channel types (0,1), eta = -2,-1 or 1,2
      //for HO0/YB0 dphi = 12 (6 phi values),  2 channel types (0,1), eta = 0
      } else{
	if (ieta<0) index      = ((iphi+1)/12-1) + 36*channel + 6*(ieta+2) + 348;
	else if (ieta>0) index = ((iphi+1)/12-1) + 36*channel + 6*(ieta+2) + 6 + 348;
	else index             = ((iphi+1)/6-1)  + 36*channel + 6*(ieta+2) + 348;
      }
    } else {
      std::cout << "HCAL Det Id not valid!" << std::endl;
      index = 0;
    }
        
  } else if (tid.calibFlavor()==HcalCalibDetId::HOCrosstalk) {
    //std::cout<<"HX:  ";
    //for YB0/HO0 phi is grouped in 6 groups of 6 with dphi=2 but the transitions are 1 or 3
    // in such a way that the %36 operation yeilds unique values for every iphi
    if (abs(ieta)==4)  index = ((iphi-1)%36) + (((zside+1)*36)/2) + 72 + 425;   //ieta = 1 YB0/HO0;
    else               index = (iphi-1) + (36*(zside+1)*2) + 425;  //ieta = 0 for HO2M/HO1M ieta=2 for HO1P/HO2P;
  }
  //std::cout << "  " << ieta << "  " << zside << "  " << iphi << "  " << depth << "  " << index << std::endl;
  return index;
}


unsigned int HcalTopology::detId2denseId(const DetId& id) const {
  unsigned int retval(0);
  if (topoVersion_==0) { // pre-LS1
    retval = detId2denseIdPreLS1(id);
  } else if (topoVersion_==10) {
    HcalDetId hid(id);
    if (hid.subdet()==HcalBarrel) {
      retval=(hid.depth()-1)+maxDepthHB_*(hid.iphi()-1);
      if (hid.ieta()>0) {
	retval+=maxDepthHB_*72*(hid.ieta()-firstHBRing());
      } else {
	retval+=maxDepthHB_*72*(hid.ieta()+lastHBRing()+nEtaHB_);
      }
    } else if (hid.subdet()==HcalEndcap) {
      retval=HBSize_;
      retval+=(hid.depth()-1)+maxDepthHE_*(hid.iphi()-1);
      if (hid.ieta()>0) {
	retval+=maxDepthHE_*72*(hid.ieta()-firstHERing());
      } else {
	retval+=maxDepthHE_*72*(hid.ieta()+lastHERing()+nEtaHE_);
      }
    } else if (hid.subdet()==HcalOuter) {
      retval=HBSize_+HESize_;
      if   (hid.ieta()>0) retval+=(hid.iphi()-1)+72*(hid.ieta()-1);
      else retval+=(hid.iphi()-1)+72*(30+hid.ieta());
    } else if (hid.subdet()==HcalForward) { 
      retval=HBSize_+HESize_+HOSize_;
      retval+=hid.depth()-1+2*(hid.iphi()-1);
      if (hid.ieta()>0) retval+=2*72*(hid.ieta()-29);
      else retval+=2*72*((41+13)+hid.ieta());
    } else {
      return 0xFFFFFFFu;
    }
  }
  return retval;
}

DetId HcalTopology::denseId2detId(unsigned int denseid) const {

  HcalSubdetector sd ( HcalBarrel ) ;
  int ie ( 0 ) ;
  int ip ( 0 ) ;
  int dp ( 0 ) ;
  int in ( denseid ) ;
  int iz ( 1 ) ;
  if (topoVersion_==0) { //DL// pre-LS1
    if (denseid < kSizeForDenseIndexingPreLS1) {
      if ( in > 2*( kHBhalf + kHEhalf + kHOhalf ) - 1 ) { // HF
	sd  = HcalForward ;
	in -= 2*( kHBhalf + kHEhalf + kHOhalf ) ; 
	iz  = ( in<kHFhalf ? 1 : -1 ) ;
	in %= kHFhalf ; 
	ip  = 4*( in/48 ) ;
	in %= 48 ;
	ip += 1 + ( in>21 ? 2 : 0 ) ;
	if( 3 == ip%4 ) in -= 22 ;
	ie  = 29 + in/2 ;
	dp  = 1 + in%2 ;
      } else if ( in > 2*( kHBhalf + kHEhalf ) - 1 ) { // HO
	sd  = HcalOuter ;
	in -= 2*( kHBhalf + kHEhalf ) ; 
	iz  = ( in<kHOhalf ? 1 : -1 ) ;
	in %= kHOhalf ; 
	dp  = 4 ;
	ip  = 1 + in/15 ;
	ie  = 1 + ( in - 15*( ip - 1 ) ) ;
      } else if ( in > 2*kHBhalf - 1 ) { // Endcap
	sd  = HcalEndcap ;
	in -= 2*kHBhalf ;
	iz  = ( in<kHEhalf ? 1 : -1 ) ;
	in %= kHEhalf ; 
	ip  = 2*( in/36 ) ;
	in %= 36 ;
	ip += 1 + in/28 ;
	if( 0 == ip%2 ) in %= 28 ;
	ie  = 15 + ( in<2 ? 1 + in : 2 + 
		     ( in<20 ? 1 + ( in - 2 )/2 : 9 +
		       ( in<26 ? 1 + ( in - 20 )/3 : 3 ) ) ) ;
	dp  = ( in<1 ? 3 :
		( in<2 ? 1 : 
		  ( in<20 ? 1 + ( in - 2 )%2 : 
		    ( in<26 ? 1 + ( in - 20 )%3 : 
		      ( 1 + ( in - 26 )%2 ) ) ) ) ) ;
      } else { // barrel
	iz  = ( in<kHBhalf ? 1 : -1 ) ;
	in %= kHBhalf ; 
	ip = in/18 + 1 ;
	in %= 18 ;
	if ( in < 14 ) {
	  dp = 1 ;
	  ie = in + 1 ;
	} else {
	  in %= 14 ;
	  dp =  1 + in%2 ;
	  ie = 15 + in/2 ;
	}
      }
    }
  } else if (topoVersion_==10) {
    if (denseid < ncells()) {
      if (denseid >= (HBSize_+HESize_+HOSize_)) {
	sd  = HcalForward ;
	in -= (HBSize_+HESize_+HOSize_);
	dp  = (in%2) + 1;
	ip  = (in - dp + 1)%144;
	ip  = (ip/2) + 1;
	ie  = (in - dp + 1 - 2*(ip -1))/144;
	if (ie > 12) {ie  = 54 -ie; iz = -1;}
	else         {ie += 29;     iz =  1;}
      } else if (denseid >= (HBSize_+HESize_)) {
	sd  = HcalOuter ;
	in -= (HBSize_+HESize_);
	dp  = 4;
	ip  = (in%72) + 1;
	ie  = (in - ip + 1)/72;
	if (ie > 14) {ie  = 30 -ie; iz = -1;}
	else         {ie += 1;      iz =  1;}
      } else if (denseid >= (HBSize_)) {
	sd  = HcalEndcap ;
	in -= (HBSize_);
	dp  = (in%maxDepthHE_)+1;
	ip  = (in - dp + 1)%(maxDepthHE_*72);
	ip  = (ip/maxDepthHE_) + 1;
	ie  = (in - dp + 1 - maxDepthHE_*(ip-1))/(72*maxDepthHE_);
	if (ie >= nEtaHE_) {ie = lastHERing()+nEtaHE_ - ie; iz = -1;}
	else               {ie = firstHERing() + ie;        iz =  1;}
      } else {
	sd  = HcalBarrel ;
	dp  = (in%maxDepthHB_)+1;
	ip  = (in - dp + 1)%(maxDepthHB_*72);
	ip  = (ip/maxDepthHB_) + 1;
	ie  = (in - dp + 1 - maxDepthHB_*(ip-1))/(72*maxDepthHB_);
	if (ie >= nEtaHB_) {ie  = lastHBRing()+nEtaHB_ - ie; iz = -1;}
	else               {ie = firstHERing() + ie;         iz =  1;}
      }	
    }
  }
  return HcalDetId( sd, iz*int(ie), ip, dp );
}

unsigned int HcalTopology::ncells() const {
  return HBSize_+HESize_+HOSize_+HFSize_;
}

int HcalTopology::topoVersion() const {
  return topoVersion_;
}
