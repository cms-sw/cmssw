//this must be changed when scramming is working
#include"L1Trigger/GlobalCaloTrigger/interface/L1GctEmCand.h"

L1GctEmCand::L1GctEmCand(ULong rank, ULong eta, ULong phi) : 
  myRank(rank),
  myEta(eta),
  myPhi(phi)
{

}

L1GctEmCand::L1GctEmCand(ULong rawData) {
    
    myRank = rawData & 0x3f;
    rawData >>= RANK_BITWIDTH;   //shift the remaining bits down, to remove the rank info         
    myPhi = rawData & 0x1;  //1 bit of Phi
    myEta = (rawData & 0xE) >> 1;  //other 3 bits are eta
  
}


L1GctEmCand::~L1GctEmCand(){
}
	
	



