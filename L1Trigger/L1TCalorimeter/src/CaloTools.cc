#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"


#include "L1Trigger/L1TCalorimeter/interface/CaloStage2Nav.h"

const l1t::CaloTower l1t::CaloTools::nullTower_;
const l1t::CaloCluster l1t::CaloTools::nullCluster_;

const float l1t::CaloTools::kGTEtaLSB = 0.0435;
const float l1t::CaloTools::kGTPhiLSB = 0.0435;
const float l1t::CaloTools::kGTEtLSB = 0.5;

const int64_t l1t::CaloTools::cos_coeff[72] = {1023, 1019, 1007, 988, 961, 927, 886, 838, 784, 723, 658, 587, 512, 432, 350, 265, 178, 89, 0, -89, -178, -265, -350, -432, -512, -587, -658, -723, -784, -838, -886, -927, -961, -988, -1007, -1019, -1023, -1019, -1007, -988, -961, -927, -886, -838, -784, -723, -658, -587, -512, -432, -350, -265, -178, -89, 0, 89, 178, 265, 350, 432, 511, 587, 658, 723, 784, 838, 886, 927, 961, 988, 1007, 1019};

const int64_t l1t::CaloTools::sin_coeff[72] = {0, 89, 178, 265, 350, 432, 512, 587, 658, 723, 784, 838, 886, 927, 961, 988, 1007, 1019, 1023, 1019, 1007, 988, 961, 927, 886, 838, 784, 723, 658, 587, 512, 432, 350, 265, 178, 89, 0, -89, -178, -265, -350, -432, -512, -587, -658, -723, -784, -838, -886, -927, -961, -988, -1007, -1019, -1023, -1019, -1007, -988, -961, -927, -886, -838, -784, -723, -658, -587, -512, -432, -350, -265, -178, -89};



bool l1t::CaloTools::insertTower(std::vector<l1t::CaloTower>& towers, const l1t::CaloTower& tower) {
  size_t towerIndex = CaloTools::caloTowerHash(tower.hwEta(), tower.hwPhi());
  if (towers.size() > towerIndex) {
    towers.at(towerIndex) = tower;
    return true;
  }
  else return false;
}

//currently implemented as a brute force search but this will hopefully change in the future
//with standarising the layout of std::vector<l1t::CaloTower>
const l1t::CaloTower& l1t::CaloTools::getTower(const std::vector<l1t::CaloTower>& towers,int iEta,int iPhi)
{
  if(abs(iEta) > CaloTools::kHFEnd) return nullTower_;

  size_t towerIndex = CaloTools::caloTowerHash(iEta, iPhi);
  if(towerIndex<towers.size()){
    if(towers[towerIndex].hwEta()!=iEta || towers[towerIndex].hwPhi()!=iPhi){ //it failed, this is bad, but we will not log the error due to policy and silently attempt to do a brute force search instead 
      //std::cout <<"error, tower "<<towers[towerIndex].hwEta()<<" "<<towers[towerIndex].hwPhi()<<" does not match "<<iEta<<" "<<iPhi<<" index "<<towerIndex<<" nr towrs "<<towers.size()<<std::endl;
      for(size_t towerNr=0;towerNr<towers.size();towerNr++){
	if(towers[towerNr].hwEta()==iEta && towers[towerNr].hwPhi()==iPhi) return towers[towerNr];
      }     
    }else return towers[towerIndex];
  
  }
  else{// in case the vector of towers do not contain all the towers (towerIndex can be > towers.size())
    for(size_t towerNr=0;towerNr<towers.size();towerNr++){
	  if(towers[towerNr].hwEta()==iEta && towers[towerNr].hwPhi()==iPhi) return towers[towerNr];
    }
  }

  return nullTower_;
}

const l1t::CaloCluster& l1t::CaloTools::getCluster(const std::vector<l1t::CaloCluster>& clusters,int iEta,int iPhi)
{
  for(size_t clusterNr=0;clusterNr<clusters.size();clusterNr++){
    if(clusters[clusterNr].hwEta()==iEta && clusters[clusterNr].hwPhi()==iPhi) return clusters[clusterNr];
  }
  return nullCluster_;
}



//this implimentation has not all the necessary info yet, we need to check the exact HF numbering
//(iEta=-28,iPhi=1)=index 0 to (iEta=28,iPhi=72)=index 28*72*2-1
//HF then runs after that so -32,1 = 28*72*2
size_t l1t::CaloTools::caloTowerHash(int iEta,int iPhi)
{

  if(!isValidIEtaIPhi(iEta,iPhi)) return caloTowerHashMax();
  else{
    const int absIEta = abs(iEta);
    if(absIEta>kHFEnd) return kNrTowers;
    else if(absIEta<=kHBHEEnd){ //HBHE
      int iEtaNoZero=iEta;
      if(iEta>0) iEtaNoZero--;
      return (iEtaNoZero+kHBHEEnd)*kHBHENrPhi+iPhi-1;
    }else{ //HF
      int iEtaIndex = iEta+kHFEnd; //iEta=-32 is 0
      if(iEta>0) iEtaIndex= iEta-kHBHEEnd+(kHFEnd-kHBHEEnd)-1; //but iEta=29 is 4
      return iEtaIndex*kHFNrPhi+iPhi/kHFPhiSeg + kNrHBHETowers;
    }
  }
}


size_t l1t::CaloTools::caloTowerHashMax()
{
  return kNrTowers;
}


bool l1t::CaloTools::isValidIEtaIPhi(int iEta,int iPhi)
{
  size_t absIEta = abs(iEta);
  if(iPhi<=0 || iPhi>kNPhi) return false;
  if(absIEta==0 || absIEta>kHFEnd) return false;
  //if(absIEta>kHBHEEnd && iPhi%kHFPhiSeg!=1) return false;
  return true;

}

int l1t::CaloTools::calHwEtSum(int iEta,int iPhi,const std::vector<l1t::CaloTower>& towers,
			       int localEtaMin,int localEtaMax,int localPhiMin,int localPhiMax,
			       SubDet etMode)
{

  return calHwEtSum(iEta,iPhi,towers,localEtaMin,localEtaMax,localPhiMin,localPhiMax,kHFEnd,etMode);
}

int l1t::CaloTools::calHwEtSum(int iEta,int iPhi,const std::vector<l1t::CaloTower>& towers,
			       int localEtaMin,int localEtaMax,int localPhiMin,int localPhiMax,
			       int iEtaAbsMax,SubDet etMode)
{
  int hwEtSum=0;
  for(int etaNr=localEtaMin;etaNr<=localEtaMax;etaNr++){
    for(int phiNr=localPhiMin;phiNr<=localPhiMax;phiNr++){
      
      int towerIEta = l1t::CaloStage2Nav::offsetIEta(iEta,etaNr);
      int towerIPhi = l1t::CaloStage2Nav::offsetIPhi(iPhi,phiNr);
      if(abs(towerIEta)<=iEtaAbsMax){
	const l1t::CaloTower& tower = getTower(towers,CaloTools::caloEta(towerIEta),towerIPhi);
	if(etMode==ECAL) hwEtSum+=tower.hwEtEm();
	else if(etMode==HCAL) hwEtSum+=tower.hwEtHad();
	else if(etMode==CALO) hwEtSum+=tower.hwPt();
      }	
    }
  }
  return hwEtSum;
}


size_t l1t::CaloTools::calNrTowers(int iEtaMin,int iEtaMax,int iPhiMin,int iPhiMax,const std::vector<l1t::CaloTower>& towers,int minHwEt,int maxHwEt,SubDet etMode)
{
  size_t nrTowers=0;
  l1t::CaloStage2Nav nav(iEtaMin,iPhiMin);
  while(nav.currIEta()<=iEtaMax){
    bool finishPhi = false;
    while(!finishPhi){
      const l1t::CaloTower& tower = l1t::CaloTools::getTower(towers,CaloTools::caloEta(nav.currIEta()),nav.currIPhi());
      int towerHwEt =0;
      if(etMode==ECAL) towerHwEt+=tower.hwEtEm();
      else if(etMode==HCAL) towerHwEt+=tower.hwEtHad();
      else if(etMode==CALO) towerHwEt+=tower.hwPt();
      if(towerHwEt>=minHwEt && towerHwEt<=maxHwEt) nrTowers++;
      finishPhi = (nav.currIPhi() == iPhiMax);
	  nav.north();
    }
    nav.east();
    nav.resetIPhi();
  }
  return nrTowers;
}

std::pair<float,float> l1t::CaloTools::towerEtaBounds(int ieta)
{
  if(ieta==0) ieta = 1;
  if(ieta>kHFEnd) ieta = kHFEnd;
  if(ieta<(-1*kHFEnd)) ieta = -1*kHFEnd;
  //const float towerEtas[33] = {0,0.087,0.174,0.261,0.348,0.435,0.522,0.609,0.696,0.783,0.870,0.957,1.044,1.131,1.218,1.305,1.392,1.479,1.566,1.653,1.740,1.830,1.930,2.043,2.172,2.322,2.5,2.650,3.000,3.5,4.0,4.5,5.0}; 
  const float towerEtas[42] = {0,0.087,0.174,0.261,0.348,0.435,0.522,0.609,0.696,0.783,0.870,0.957,1.044,1.131,1.218,1.305,1.392,1.479,1.566,1.653,1.740,1.830,1.930,2.043,2.172,2.322,2.5,2.650,2.853,3.139,3.314,3.489,3.664,3.839,4.013,4.191,4.363,4.538,4.716,4.889,5.191,5.191};
  return std::make_pair( towerEtas[abs(ieta)-1],towerEtas[abs(ieta)] );
}

float l1t::CaloTools::towerEta(int ieta)
{
  std::pair<float,float> bounds = towerEtaBounds(ieta);
  float eta = (bounds.second+bounds.first)/2.;
  float sign = ieta>0 ? 1. : -1.;
  return sign*eta; 
}

float l1t::CaloTools::towerPhi(int ieta, int iphi)
{
  float phi = (float(iphi)-0.5)*towerPhiSize(ieta);
  if (phi > M_PI) phi = phi - (2*M_PI);
  return phi;
}

float l1t::CaloTools::towerEtaSize(int ieta)
{
  std::pair<float,float> bounds = towerEtaBounds(ieta);
  float size = (bounds.second-bounds.first);
  return size;
}

float l1t::CaloTools::towerPhiSize(int ieta)
{
  return 2.*M_PI/kNPhi;
}


// convert from calo ieta to internal MP ieta
int l1t::CaloTools::mpEta(int ieta) {

  if (ieta>kHFBegin) return ieta-1;
  else if (ieta<-1*kHFBegin) return ieta+1;
  else return ieta;

}


// convert from internal MP ieta to calo ieta
int l1t::CaloTools::caloEta(int mpEta) {

  if (mpEta>=kHFBegin) return mpEta+1;
  else if (mpEta<=-1*kHFBegin) return mpEta-1;
  else return mpEta;

}


// convert calorimeter ieta to RCT region index
int l1t::CaloTools::regionEta(int ieta)
{

  // outside HF
  if (abs(ieta) > kHFEnd)
    return (ieta<0 ? 0 : 21);

  // inside HBHE
  if (abs(ieta) <= kHFBegin)
    {
      if (ieta<0)
	return 11 - ceil( double (abs(ieta) /4.) );
      else
	return ceil( double (abs(ieta) /4.) ) + 10;
    }

  // in HF
  if (ieta<0)
    return 4 - ceil( double (abs(ieta)-29) /4. );
  else
    return ceil( double (abs(ieta)-29) /4. ) + 17;

}


// convert calorimeter ieta to etaBin16 index
int l1t::CaloTools::bin16Eta(int ieta)
{
  int absIEta = abs(ieta);

  if (absIEta>0 && absIEta<=5) return 0;
  else if (absIEta<=9) return 1;
  else if (absIEta<=13) return 2;
  else if (absIEta<=15) return 3;
  else if (absIEta<=17) return 4;
  else if (absIEta<=19) return 5;
  else if (absIEta<=21) return 6;
  else if (absIEta==22) return 7;
  else if (absIEta==23) return 8;
  else if (absIEta==24) return 9;
  else if (absIEta==25) return 10;
  else if (absIEta==26) return 11;
  else if (absIEta<=28) return 12;
  else if (absIEta<=32) return 13;
  else if (absIEta<=36) return 14;
  else if (absIEta<=41) return 15;
  else return -1; // error
}


int l1t::CaloTools::gtEta(int ieta) {

  double eta = towerEta(ieta);
  return round ( eta / kGTEtaLSB );

}

int l1t::CaloTools::gtPhi(int ieta, int iphi) {

  double phi = towerPhi(ieta, iphi);
  if (phi<0) phi = phi + 2*M_PI;
  return round ( phi / kGTPhiLSB );

}





// this conversion is based on GT input definitions in CMS DN-2014/029 
math::PtEtaPhiMLorentzVector l1t::CaloTools::p4Demux(l1t::L1Candidate* cand) {

  return math::PtEtaPhiMLorentzVector( cand->hwPt() * kGTEtLSB + 1.E-6,
				       cand->hwEta() * kGTEtaLSB,
				       cand->hwPhi() * kGTPhiLSB,
				       0. ) ;
  
}


l1t::EGamma l1t::CaloTools::egP4Demux(l1t::EGamma& eg) {
  
 l1t::EGamma tmpEG( p4Demux(&eg),
  		      eg.hwPt(),
  		      eg.hwEta(),
  		      eg.hwPhi(),
  		      eg.hwQual(),
  		      eg.hwIso() );
 tmpEG.setTowerIPhi(eg.towerIPhi());
 tmpEG.setTowerIEta(eg.towerIEta());
 tmpEG.setRawEt(eg.rawEt());
 tmpEG.setIsoEt(eg.isoEt());
 tmpEG.setFootprintEt(eg.footprintEt());
 tmpEG.setNTT(eg.nTT());
 tmpEG.setShape(eg.shape());
 tmpEG.setTowerHoE(eg.towerHoE());

 return tmpEG;

}


l1t::Tau l1t::CaloTools::tauP4Demux(l1t::Tau& tau) {

  l1t::Tau tmpTau ( p4Demux(&tau),
		    tau.hwPt(),
		    tau.hwEta(),
		    tau.hwPhi(),
		    tau.hwQual(),
		    tau.hwIso());
  tmpTau.setTowerIPhi(tau.towerIPhi());
  tmpTau.setTowerIEta(tau.towerIEta());
  tmpTau.setRawEt(tau.rawEt());
  tmpTau.setIsoEt(tau.isoEt());
  tmpTau.setNTT(tau.nTT());
  tmpTau.setHasEM(tau.hasEM());
  tmpTau.setIsMerged(tau.isMerged());

  return tmpTau;

}


l1t::Jet l1t::CaloTools::jetP4Demux(l1t::Jet& jet) {

  
  l1t::Jet tmpJet ( p4Demux(&jet),
		   jet.hwPt(),
		   jet.hwEta(),
		   jet.hwPhi(),
		   jet.hwQual() );
  tmpJet.setTowerIPhi(jet.towerIPhi());
  tmpJet.setTowerIEta(jet.towerIEta());
  tmpJet.setRawEt(jet.rawEt());
  tmpJet.setSeedEt(jet.seedEt());
  tmpJet.setPUEt(jet.puEt());
  tmpJet.setPUDonutEt(0,jet.puDonutEt(0));
  tmpJet.setPUDonutEt(1,jet.puDonutEt(1));
  tmpJet.setPUDonutEt(2,jet.puDonutEt(2));
  tmpJet.setPUDonutEt(3,jet.puDonutEt(3));

  return tmpJet;
  
}


l1t::EtSum l1t::CaloTools::etSumP4Demux(l1t::EtSum& etsum) {

  return l1t::EtSum( p4Demux(&etsum),
		     etsum.getType(),
		     etsum.hwPt(),
		     etsum.hwEta(),
		     etsum.hwPhi(),
		     etsum.hwQual() );
  
}



// 
math::PtEtaPhiMLorentzVector l1t::CaloTools::p4MP(l1t::L1Candidate* cand) {

  return math::PtEtaPhiMLorentzVector( cand->hwPt() * 0.5 + 1.E-6,
				       towerEta(cand->hwEta()),
				       towerPhi(cand->hwEta(), cand->hwPhi()),
				       0. ) ;

}

l1t::EGamma l1t::CaloTools::egP4MP(l1t::EGamma& eg) {

  l1t::EGamma tmpEG( p4MP(&eg),
		     eg.hwPt(),
		     eg.hwEta(),
		     eg.hwPhi(),
		     eg.hwQual(),
		     eg.hwIso() );
  tmpEG.setTowerIPhi(eg.towerIPhi());
  tmpEG.setTowerIEta(eg.towerIEta());
  tmpEG.setRawEt(eg.rawEt());
  tmpEG.setIsoEt(eg.isoEt());
  tmpEG.setFootprintEt(eg.footprintEt());
  tmpEG.setNTT(eg.nTT());
  tmpEG.setShape(eg.shape());
  
  return tmpEG;

}


l1t::Tau l1t::CaloTools::tauP4MP(l1t::Tau& tau) {

  l1t::Tau tmpTau ( p4MP(&tau),
		    tau.hwPt(),
		    tau.hwEta(),
		    tau.hwPhi(),
		    tau.hwQual(),
		    tau.hwIso());
  tmpTau.setTowerIPhi(tau.towerIPhi());
  tmpTau.setTowerIEta(tau.towerIEta());
  tmpTau.setRawEt(tau.rawEt());
  tmpTau.setIsoEt(tau.isoEt());
  tmpTau.setNTT(tau.nTT());
  tmpTau.setHasEM(tau.hasEM());
  tmpTau.setIsMerged(tau.isMerged());

  return tmpTau;
}


l1t::Jet l1t::CaloTools::jetP4MP(l1t::Jet& jet) {

  l1t::Jet tmpJet ( p4MP(&jet),
		   jet.hwPt(),
		   jet.hwEta(),
		   jet.hwPhi(),
		   jet.hwQual() );
  tmpJet.setTowerIPhi(jet.towerIPhi());
  tmpJet.setTowerIEta(jet.towerIEta());
  tmpJet.setRawEt(jet.rawEt());
  tmpJet.setSeedEt(jet.seedEt());
  tmpJet.setPUEt(jet.puEt());
  tmpJet.setPUDonutEt(0,jet.puDonutEt(0));
  tmpJet.setPUDonutEt(1,jet.puDonutEt(1));
  tmpJet.setPUDonutEt(2,jet.puDonutEt(2));
  tmpJet.setPUDonutEt(3,jet.puDonutEt(3));

  return tmpJet;

}

l1t::EtSum l1t::CaloTools::etSumP4MP(l1t::EtSum& etsum) {

  return l1t::EtSum( p4MP(&etsum),
		     etsum.getType(),
		     etsum.hwPt(),
		     etsum.hwEta(),
		     etsum.hwPhi(),
		     etsum.hwQual() );
  
}
