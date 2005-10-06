#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include <cmath>
#include <iostream>


  static const int IPHI_MAX=72;

  HcalTopology::HcalTopology() :
    firstHBRing_(1),
    lastHBRing_(16),
    firstHERing_(16),
    lastHERing_(29),
    firstHFRing_(29),
    lastHFRing_(41),
    firstHORing_(1),
    lastHORing_(15),
    firstHEDoublePhiRing_(21),
    firstHFQuadPhiRing_(40),
    firstHETripleDepthRing_(27),
    singlePhiBins_(72),
    doublePhiBins_(36)
{
    min_iphi_=1;
    max_iphi_=IPHI_MAX;
    min_ieta_=-41;
    max_ieta_=41;
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
     ieta=40-41 have 18 in iphi
     all have two depths

  HO has 15 towers in ieta and 72 in iphi and depth = 4 (one value)
  */

  /** Is this a valid cell id? */
  bool HcalTopology::valid(const HcalDetId& id) const {
    bool ok=true;
    int ieta=id.ieta();
    int aieta=id.ietaAbs();
    int depth=id.depth();
    int iphi=id.iphi();

    if ((ieta==0 || iphi<=0 || depth<=0 || iphi>IPHI_MAX) ||
	(ieta<min_ieta_ || ieta>max_ieta_ || iphi>max_iphi_ || iphi<min_iphi_)) ok=false;
    
    if (ok) {
      HcalSubdetector subdet=id.subdet();
      if (subdet==HcalBarrel) {
	if (aieta>16 || depth>2 || (aieta<=14 && depth>1)) ok=false;	    
      } else if (subdet==HcalEndcap) {
	if (aieta<16 || aieta>29 ||
	    (aieta==16 && depth!=3) ||
	    (aieta==17 && depth!=1) ||
	    (((aieta>=18 && aieta<=26) || aieta==29) && depth>2) ||
	    ((aieta==27 || aieta==28) && depth>3) ||
	    (aieta>=21 && (iphi%2)==0)) ok=false;
      } else if (subdet==HcalOuter) {
	if (aieta>15 || iphi>IPHI_MAX || depth!=4) ok=false;
      } else if (subdet==HcalForward) {
	if (aieta<29 || aieta>41 ||
	    ((iphi%2)==0) ||
	    (depth>2) ||
	    (aieta>=40 && ((iphi+1)%4)==0)) ok=false;
      } else ok=false;
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
	else neighbor=HcalDetId(id.subdet(),id.ieta(),id.iphi()+1,id.depth()); 
	if (neighbor.iphi()>max_iphi_ || neighbor.iphi()<min_iphi_) ok=false;
	break;
      case (HcalEndcap):
	if (id.ietaAbs()>=21) {
	  if (id.iphi()==IPHI_MAX-1) neighbor=HcalDetId(HcalEndcap,id.ieta(),1,id.depth()); 
	  else neighbor=HcalDetId(HcalEndcap,id.ieta(),id.iphi()+2,id.depth()); 
	} else {
	  if (id.iphi()==IPHI_MAX) neighbor=HcalDetId(HcalEndcap,id.ieta(),1,id.depth()); 
	  else neighbor=HcalDetId(HcalEndcap,id.ieta(),id.iphi()+1,id.depth()); 
	}
	if (neighbor.iphi()>max_iphi_ || neighbor.iphi()<min_iphi_) ok=false;
	break;
      case (HcalForward):
	if (id.ietaAbs()>=40) {
	  if (id.iphi()==IPHI_MAX-3) neighbor=HcalDetId(HcalEndcap,id.ieta(),1,id.depth()); 
	  else neighbor=HcalDetId(HcalEndcap,id.ieta(),id.iphi()+4,id.depth()); 
	} else {
	  if (id.iphi()==IPHI_MAX-1) neighbor=HcalDetId(HcalEndcap,id.ieta(),1,id.depth()); 
	  else neighbor=HcalDetId(HcalEndcap,id.ieta(),id.iphi()+2,id.depth()); 
	}
	if (neighbor.iphi()>max_iphi_ || neighbor.iphi()<min_iphi_) ok=false;
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
	else neighbor=HcalDetId(id.subdet(),id.ieta(),id.iphi()-1,id.depth()); 
	if (neighbor.iphi()>max_iphi_ || neighbor.iphi()<min_iphi_) ok=false;
	break;
      case (HcalEndcap):
	if (id.ietaAbs()>=21) {
	  if (id.iphi()==1) neighbor=HcalDetId(HcalEndcap,id.ieta(),IPHI_MAX-1,id.depth()); 
	  else neighbor=HcalDetId(HcalEndcap,id.ieta(),id.iphi()-2,id.depth()); 
	} else {
	  if (id.iphi()==1) neighbor=HcalDetId(HcalEndcap,id.ieta(),IPHI_MAX,id.depth()); 
	  else neighbor=HcalDetId(HcalEndcap,id.ieta(),id.iphi()-1,id.depth()); 
	}
	if (neighbor.iphi()>max_iphi_ || neighbor.iphi()<min_iphi_) ok=false;
	break;
      case (HcalForward):
	if (id.ietaAbs()>=40) {
	  if (id.iphi()==1) neighbor=HcalDetId(HcalEndcap,id.ieta(),IPHI_MAX-3,id.depth()); 
	  else neighbor=HcalDetId(HcalEndcap,id.ieta(),id.iphi()-4,id.depth()); 
	} else {
	  if (id.iphi()==1) neighbor=HcalDetId(HcalEndcap,id.ieta(),IPHI_MAX-1,id.depth()); 
	  else neighbor=HcalDetId(HcalEndcap,id.ieta(),id.iphi()-2,id.depth()); 
	}
	if (neighbor.iphi()>max_iphi_ || neighbor.iphi()<min_iphi_) ok=false;
	break;
      default: ok=false;
      }
    } 
    return ok;
  }

  int HcalTopology::incIEta(const HcalDetId& id, HcalDetId neighbors[2]) const {
    if (id.zside()==1) return incAIEta(id,neighbors);
    else return decAIEta(id,neighbors);
  }

  int HcalTopology::decIEta(const HcalDetId& id, HcalDetId neighbors[2]) const {
    if (id.zside()==1) return decAIEta(id,neighbors);
    else return incAIEta(id,neighbors);
  }

  /** Increasing in |ieta|, there is always at most one neighbor */
  int HcalTopology::incAIEta(const HcalDetId& id, HcalDetId neighbors[2]) const {
    int n=1;
    int aieta=id.ietaAbs();

    if (aieta==20 && (id.iphi()%2)==0) 
      neighbors[0]=HcalDetId(id.subdet(),(aieta+1)*id.zside(),id.iphi()-1,id.depth());
    else if (aieta==39 && ((id.iphi()+1)%4)==0) 
      neighbors[0]=HcalDetId(id.subdet(),(aieta+1)*id.zside(),id.iphi()-2,id.depth());
    else
      neighbors[0]=HcalDetId(id.subdet(),(aieta+1)*id.zside(),id.iphi(),id.depth());
    
    if (!valid(neighbors[0])) n=0;
    return n;
  }

  /** Decreasing in |ieta|, there are be two neighbors of 40 and 21*/
  int HcalTopology::decAIEta(const HcalDetId& id, HcalDetId neighbors[2]) const {
    int n=1;
    int aieta=id.ietaAbs();

    if (aieta==21) { 
      n=2;
      neighbors[0]=HcalDetId(id.subdet(),(aieta-1)*id.zside(),id.iphi(),id.depth());
      neighbors[1]=HcalDetId(id.subdet(),(aieta-1)*id.zside(),id.iphi()+1,id.depth());
    } else if (aieta==40) {
      n=2;
      neighbors[0]=HcalDetId(id.subdet(),(aieta-1)*id.zside(),id.iphi(),id.depth());
      neighbors[1]=HcalDetId(id.subdet(),(aieta-1)*id.zside(),id.iphi()+2,id.depth());
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

    return n;
  }


void HcalTopology::depthBinInformation(HcalSubdetector subdet, int etaRing,
                                       int & nDepthBins, int & startingBin) const {
  if(subdet == HcalBarrel) {
    if (etaRing<=14) {
      nDepthBins = 1;
      startingBin = 1;
    } else {
      nDepthBins = 2;
      startingBin = 1;
    }
  } else if(subdet == HcalEndcap) {
    if (etaRing==16) {
      nDepthBins = 1;
      startingBin = 3;
    } else if (etaRing==17) {
      nDepthBins = 1;
      startingBin = 1;
    } else {
      nDepthBins = (etaRing >= firstHETripleDepthRing_) ? 3 : 2;
      startingBin = 1;
    }
  }

  else if(subdet == HcalForward) {
    nDepthBins = 2;
    startingBin = 1;
  }

  else if(subdet == HcalOuter) {
    nDepthBins = 1;
    startingBin = 4;
  }

  else {
    std::cerr << "Bad HCAL subdetector " << subdet << std::endl;
  }
}


int HcalTopology::nPhiBins(int etaRing) const {
  int lastPhiBin = (etaRing < firstHEDoublePhiRing_) ? singlePhiBins_ : doublePhiBins_;
  return std::min(lastPhiBin, max_iphi_) - std::max(min_iphi_, 1) + 1;
}



