#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"


#include "L1Trigger/L1TCalorimeter/interface/CaloStage2Nav.h"

const l1t::CaloTower l1t::CaloTools::nullTower_;
const l1t::CaloCluster l1t::CaloTools::nullCluster_;

const float l1t::CaloTools::kGTEtaLSB = 0.0435;
const float l1t::CaloTools::kGTPhiLSB = 0.0435;
const float l1t::CaloTools::kGTEtLSB = 0.5;

const int64_t l1t::CaloTools::cos_coeff[72] = {1023, 1019, 1007, 988, 961, 927, 886, 838, 784, 723, 658, 587, 512, 432, 350, 265, 178, 89, 0, -89, -178, -265, -350, -432, -512, -587, -658, -723, -784, -838, -886, -927, -961, -988, -1007, -1019, -1023, -1019, -1007, -988, -961, -927, -886, -838, -784, -723, -658, -587, -512, -432, -350, -265, -178, -89, 0, 89, 178, 265, 350, 432, 511, 587, 658, 723, 784, 838, 886, 927, 961, 988, 1007, 1019};

const int64_t l1t::CaloTools::sin_coeff[72] = {0, 89, 178, 265, 350, 432, 512, 587, 658, 723, 784, 838, 886, 927, 961, 988, 1007, 1019, 1023, 1019, 1007, 988, 961, 927, 886, 838, 784, 723, 658, 587, 512, 432, 350, 265, 178, 89, 0, -89, -178, -265, -350, -432, -512, -587, -658, -723, -784, -838, -886, -927, -961, -988, -1007, -1019, -1023, -1019, -1007, -988, -961, -927, -886, -838, -784, -723, -658, -587, -512, -432, -350, -265, -178, -89};

// mapping between sums in emulator and data
const int l1t::CaloTools::emul_to_data_sum_index_map[31] = {
  9, 1, 19, 8, 0, 18, 10, 4, 6, 14,     // 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
  28, 24, 13, 27, 23, 15, 5, 7, 22, 12, // 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
  3, 21, 11, 2, 20, 17, 30, 26, 16, 29, // 20, 21, 22, 23, 24, 25, 26, 27, 28, 29
  25                                    // 30, 31
};


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
	const l1t::CaloTower& tower = getTower(towers,towerIEta,towerIPhi);
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
  tmpEG.setTowerHoE(eg.towerHoE());
  
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
unsigned int l1t::CaloTools::gloriousDivision( uint32_t aNumerator , uint32_t aDenominator)
{ 

  static const uint64_t lLut[] = { 0, 
           16777215, 4194304, 1864135, 1048576, 671089, 466034, 342392, 262144, 207126, 167772, 138655, 116508,  99273,  85598,  74565,
	 65536,  58053,  51782,  46474,  41943,  38044,  34664,  31715,  29127,  26844,  24818,  23014,  21400,  19949,  18641,  17458,
	 16384,  15406,  14513,  13696,  12945,  12255,  11619,  11030,  10486,   9980,   9511,   9074,   8666,   8285,   7929,   7595,
	  7282,   6988,   6711,   6450,   6205,   5973,   5754,   5546,   5350,   5164,   4987,   4820,   4660,   4509,   4365,   4227,
	  4096,   3971,   3852,   3737,   3628,   3524,   3424,   3328,   3236,   3148,   3064,   2983,   2905,   2830,   2758,   2688,
	  2621,   2557,   2495,   2435,   2378,   2322,   2268,   2217,   2166,   2118,   2071,   2026,   1982,   1940,   1899,   1859,
	  1820,   1783,   1747,   1712,   1678,   1645,   1613,   1581,   1551,   1522,   1493,   1465,   1438,   1412,   1387,   1362,
	  1337,   1314,   1291,   1269,   1247,   1226,   1205,   1185,   1165,   1146,   1127,   1109,   1091,   1074,   1057,   1040,
	  1024,   1008,    993,    978,    963,    948,    934,    921,    907,    894,    881,    868,    856,    844,    832,    820,
	   809,    798,    787,    776,    766,    756,    746,    736,    726,    717,    707,    698,    689,    681,    672,    664,
	   655,    647,    639,    631,    624,    616,    609,    602,    594,    587,    581,    574,    567,    561,    554,    548,
	   542,    536,    530,    524,    518,    512,    506,    501,    496,    490,    485,    480,    475,    470,    465,    460,
	   455,    450,    446,    441,    437,    432,    428,    424,    419,    415,    411,    407,    403,    399,    395,    392,
	   388,    384,    380,    377,    373,    370,    366,    363,    360,    356,    353,    350,    347,    344,    340,    337,
	   334,    331,    328,    326,    323,    320,    317,    314,    312,    309,    306,    304,    301,    299,    296,    294,
	   291,    289,    286,    284,    282,    280,    277,    275,    273,    271,    268,    266,    264,    262,    260,    258,
	   256,    254,    252,    250,    248,    246,    244,    243,    241,    239,    237,    235,    234,    232,    230,    228,
	   227,    225,    223,    222,    220,    219,    217,    216,    214,    212,    211,    209,    208,    207,    205,    204,
	   202,    201,    199,    198,    197,    195,    194,    193,    191,    190,    189,    188,    186,    185,    184,    183,
	   182,    180,    179,    178,    177,    176,    175,    173,    172,    171,    170,    169,    168,    167,    166,    165,
	   164,    163,    162,    161,    160,    159,    158,    157,    156,    155,    154,    153,    152,    151,    150,    149,
	   149,    148,    147,    146,    145,    144,    143,    143,    142,    141,    140,    139,    139,    138,    137,    136,
	   135,    135,    134,    133,    132,    132,    131,    130,    129,    129,    128,    127,    127,    126,    125,    125,
	   124,    123,    123,    122,    121,    121,    120,    119,    119,    118,    117,    117,    116,    116,    115,    114,
	   114,    113,    113,    112,    111,    111,    110,    110,    109,    109,    108,    108,    107,    106,    106,    105,
	   105,    104,    104,    103,    103,    102,    102,    101,    101,    100,    100,     99,     99,     98,     98,     97,
	    97,     96,     96,     96,     95,     95,     94,     94,     93,     93,     92,     92,     92,     91,     91,     90,
	    90,     89,     89,     89,     88,     88,     87,     87,     87,     86,     86,     85,     85,     85,     84,     84,
	    84,     83,     83,     82,     82,     82,     81,     81,     81,     80,     80,     80,     79,     79,     79,     78,
	    78,     78,     77,     77,     77,     76,     76,     76,     75,     75,     75,     74,     74,     74,     73,     73,
	    73,     73,     72,     72,     72,     71,     71,     71,     70,     70,     70,     70,     69,     69,     69,     68,
	    68,     68,     68,     67,     67,     67,     67,     66,     66,     66,     66,     65,     65,     65,     65,     64 };

// Firmware uses 18bit integers - make sure we are in the same range
  aNumerator &= 0x3FFFF;
  aDenominator &= 0x3FFFF;  
  
// Shift the denominator to optimise the polynomial expansion
// I limit the shift to half the denominator size in the firmware to save on resources
  uint32_t lBitShift(0);
  for( ;lBitShift!=9 ; ++lBitShift )
  {
    if ( aDenominator & 0x20000 ) break; // There is a 1 in the MSB
    aDenominator <<= 1;
  }

// The magical polynomial expansion Voodoo
  uint64_t lInverseDenominator( ( ( aDenominator & 0x3FE00 ) - ( aDenominator & 0x001FF ) ) * ( lLut[ aDenominator >> 9 ] ) );

// Save on DSPs by throwing away a bunch of LSBs
  lInverseDenominator >>= 17;

// Multiply the numerator by the inverse denominator
  uint64_t lResult( aNumerator * lInverseDenominator );

// Add two bits to the result, to make the Voodoo below work (saves us having an if-else on the shift direction)
  lResult <<= 2;

// Restore the scale by taking into account the bitshift applied above.
// We are now 18 bit left-shifted, so the 18 LSBs are effectively the fractional part...
  
  uint32_t aFractional = ( lResult >>= ( 9 - lBitShift) ) & 0x3FFFF; 
// ...and the top 18 bits are the integer part
  
  // uint32_t aInteger    = ( lResult >>= 18 ) & 0x3FFFF; 

  unsigned int result = aFractional >> 10;

  return result;

// Simples!      
}
