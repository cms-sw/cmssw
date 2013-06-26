#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTORCAMap.h"

#include <vector>
using std::vector;

L1RCTORCAMap::L1RCTORCAMap(){
  rawEMET = std::vector<unsigned short>(56*72);
  rawHDET = std::vector<unsigned short>(56*72);
  rawEMFG = std::vector<unsigned short>(56*72);
  rawHDFG = std::vector<unsigned short>(56*72);
  rawHFET = std::vector<unsigned short>(8*18);
  combEM = std::vector<unsigned short>(56*72);
  combHD = std::vector<unsigned short>(56*72);

  barrelData = std::vector<std::vector<std::vector<unsigned short> > >(18,std::vector<std::vector<unsigned short> >(7,
							   std::vector<unsigned short>(64)));
  hfData = std::vector<std::vector<unsigned short> >(18,std::vector<unsigned short>(8));
  
}
vector<std::vector<std::vector<unsigned short> > > L1RCTORCAMap::giveBarrel(){
  return barrelData;
}

vector<std::vector<unsigned short> > L1RCTORCAMap::giveHF(){
  return hfData;
}
void L1RCTORCAMap::makeHFData(){
  std::vector<unsigned short> crate(16);
  for(int i = 0; i< 9; i++){
    for(int j = 0; j<16; j++){
      crate.at(j) = rawHFET.at(16*i+j);
    }
    hfData.at(i) = crate;
  }  
}  

void L1RCTORCAMap::makeBarrelData(){
  std::vector<int> indices;
  for(int phi = 0; phi<72;phi++){
    for(int eta = 0; eta<56;eta++){
      indices = orcamap(eta,phi);
      (barrelData.at(indices.at(0))).at(indices.at(1)).at(indices.at(2)) = combEM.at(phi*56 + eta);
      (barrelData.at(indices.at(0))).at(indices.at(1)).at(indices.at(2)+32) = combHD.at(phi*56 + eta);
    }
  }
}

void L1RCTORCAMap::readData(const std::vector<unsigned>& emet, const std::vector<unsigned>&  hdet,
			    const std::vector<bool>& emfg, const std::vector<bool>& hdfg,
			    const std::vector<unsigned>& hfet){
  for(int i = 0; i<4032; i++){
    rawEMET.at(i) = emet.at(i);
    rawHDET.at(i) = hdet.at(i);
    rawEMFG.at(i) = emfg.at(i);
    rawHDFG.at(i) = hdfg.at(i);
  }
  for(int i = 0; i<144; i++)
    rawHFET.at(i) = hfet.at(i);

  combEM = combVec(rawEMET,rawEMFG);
  combHD = combVec(rawHDET,rawHDFG);

  makeBarrelData();
  makeHFData();
}

unsigned short L1RCTORCAMap::combine(unsigned short et, unsigned short fg){
  unsigned short newfg = fg << 8;
  return newfg + et;
}

vector<unsigned short> L1RCTORCAMap::combVec(const std::vector<unsigned short>& et, 
					     const std::vector<unsigned short>& fg){
  std::vector<unsigned short> comb(56*72);
  for(int i = 0; i<(int)et.size(); i++)
    comb.at(i) = combine(et.at(i),fg.at(i));
  return comb;
}

vector<int> L1RCTORCAMap::orcamap(int eta, int phi){
  int crateNum(20);
  std::vector<int> cardTower(2,0);
  std::vector<int> returnVec(3,0);
  int modEta = eta%28;
  int modPhi = phi%8;
  if(phi < 8)
    crateNum = 0;  
  else if( phi < 16)
    crateNum = 1;
  else if( phi < 24)
    crateNum = 2;
  else if( phi < 32)
    crateNum = 3;
  else if( phi < 40)
    crateNum = 4;
  else if( phi < 48)
    crateNum = 5;
  else if( phi < 56)
    crateNum = 6;
  else if( phi < 64)
    crateNum = 7;
  else if( phi < 72)
    crateNum = 8;

  if(eta < 28)
    cardTower = lowEtaMap(modEta,modPhi);
  else {
    cardTower = highEtaMap(modEta,modPhi);
    crateNum = crateNum + 9;
  }

  returnVec.at(0) = crateNum;
  for(int i =0; i<2; i++){
    returnVec.at(i+1) = cardTower.at(i);
  }
  return returnVec;
}
    

vector<int> L1RCTORCAMap::lowEtaMap(int eta, int phi){
  int cardnum = 0;
  int towernum = 0;
  std::vector<int> returnVec(2);
  if(eta < 4){
    cardnum = 6;
    if(phi < 4)
      towernum = (3-eta)*4 + phi;
    else
      towernum = eta*4 + phi + 12;
  }
  else if(eta <12){
    if(phi < 4){
      cardnum = 4;
      if(eta < 8)
	towernum = (7-eta)*4+phi+16;
      else
	towernum = (11-eta)*4+phi;
    }
    else{
      cardnum = 5;
      if(eta < 8)
	towernum = (7-eta)*4+(phi-4)+16;
      else
	towernum = (11-eta)*4+(phi-4);
    }
  }
  else if(eta < 20){
    if(phi < 4){
      cardnum = 2;
      if(eta < 16)
	towernum = (15-eta)*4+phi+16;
      else
	towernum = (19-eta)*4+phi;
    }
    else{
      cardnum = 3;
      if(eta < 16)
	towernum = (15-eta)*4+(phi-4)+16;
      else
	towernum = (19-eta)*4+(phi-4);
    }
  }
  else if(eta < 28){
    if(phi < 4){
      cardnum = 0;
      if(eta < 24)
	towernum = (23-eta)*4+phi+16;
      else
	towernum = (27-eta)*4+phi;
    }
    else{
      cardnum = 1;
      if(eta < 24)
	towernum = (23-eta)*4+(phi-4)+16;
      else
	towernum = (27-eta)*4+(phi-4);
    }
  }
  
  returnVec.at(0) = cardnum;
  returnVec.at(1) = towernum;
  return returnVec;
}


vector<int> L1RCTORCAMap::highEtaMap(int eta, int phi){
  int cardnum;
  int towernum;
  std::vector<int> returnVec(2);
  if(eta < 8){
    if(phi < 4){
      cardnum = 0;
      towernum = eta*4+phi;
    }
    else{
      cardnum = 1;
      towernum = eta*4+(phi-4);
    }
  }
  else if(eta < 16){
    if(phi < 4){
      cardnum = 2;
      towernum = (eta-8)*4+phi;
    }
    else{
      cardnum = 3;
      towernum = (eta-8)*4+(phi-4);
    }
  }
  else if(eta < 24){
    if(phi < 4){
      cardnum = 4;
      towernum = (eta-16)*4+phi;
    }
    else{
      cardnum = 5;
      towernum = (eta-16)*4+(phi-4);
    }
  }
  else{
    cardnum = 6;
    if(phi < 4)
      towernum = (27-eta)*4+phi;
    else
      towernum = (27-eta)*4+phi+12;
  }

  returnVec.at(0)=cardnum;
  returnVec.at(1)=towernum;
  return returnVec;
}
