//#include "Utilities/Configuration/interface/Architecture.h"
#include "CalibTracker/SiStripConnectivity/interface/ModuleConnection.h"
#include <algorithm>
using namespace std;

ModuleConnection::ModuleConnection(){
  apvPairs.clear();
  moduleId = "-1";
  dcuId    = "0";
  partId   = "0";

}


ModuleConnection::ModuleConnection(VectorType v){
  apvPairs = v;
  moduleId ="-1";
  dcuId    = "0";
  partId   = "0";

  //  cout <<" GOR oo "<<apvPairs.size()<<endl;
  if (consistencyChecks() == false){
    cout <<" ModuleConnection Problem - reverting to old one."<<endl;
    apvPairs.clear();
  }
}


void ModuleConnection::setApvPairs(VectorType a){
  VectorType temp = apvPairs;
  apvPairs = a;
  if (consistencyChecks() == false){
    cout <<" ModuleConnection Problem - reverting to old one."<<endl;
    apvPairs = temp;
  }
}

void ModuleConnection::addApvPair(ApvPairConnection a){
  //
  // no check here!
  //
  apvPairs.push_back(a);
}


bool ModuleConnection::consistencyChecks(){
  if (apvPairs.size()<1 || apvPairs.size()>3){
    cout <<" Wrong number of ApvPairs "<<apvPairs.size()<<endl;
    return false;
  }    
  //
  // sort them
  //
  sort(apvPairs.begin(), apvPairs.end(), ApvI2CAddressLess());
  //
  // check that these are different
  //
  if ( (apvPairs.size() == 2 &&
	(apvPairs[0].getI2CAddressApv1() != 32 || 
	 apvPairs[1].getI2CAddressApv1() != 36) ) ||
       (apvPairs.size() == 3 &&
	(apvPairs[0].getI2CAddressApv1() != 32 || 
	 apvPairs[1].getI2CAddressApv1() != 34 || 
	 apvPairs[2].getI2CAddressApv1() != 36) ) ) {
    cout <<" Wrong I2C Addresses "<<endl;
    return false;
  }
  //
  // check that all the other parameters are fine
  //
  for (unsigned int k=1; k<apvPairs.size() ; k++){
    if ( apvPairs[k].getFecSlot() !=  apvPairs[0].getFecSlot()  ||
	 apvPairs[k].getRingSlot() !=  apvPairs[0].getRingSlot()  ||
	 apvPairs[k].getCcuAddress() !=  apvPairs[0].getCcuAddress()  ||
	 apvPairs[k].getI2CChannel() !=  apvPairs[0].getI2CChannel() ) {
      cout <<" The ApvPairConnections do NOT have same slot, ring, etc... "<<endl;
      return false;
    }
  }
  return true;

}

int ModuleConnection::getCcuAddress(){
  if (consistencyChecks() == false) return -1;
  if (apvPairs.size() == 0) return -1;

  return apvPairs[0].getCcuAddress();
  
}
int ModuleConnection::getI2CChannel(){
  if (consistencyChecks() == false) return -1;
  if (apvPairs.size() == 0) return -1;

  return apvPairs[0].getI2CChannel();
  
}


set<int> ModuleConnection::getConnectedFeds(){
  set<int> out;
  for (VectorType::iterator it = apvPairs.begin(); it != apvPairs.end(); it++){
    out.insert( (*it).getFedNumber());
  }
  return out;
}

void ModuleConnection::print() {
  cout << " Module ID " << moduleId << " Part ID " << partId 
       << " connected to CCU # " << getCcuAddress()
       << " I2C channel " << getI2CChannel() << " with  : " << endl;
  for (VectorType::iterator it = apvPairs.begin(); it != apvPairs.end(); it++){
    cout << "                         APV Pairs " 
         << (*it).getI2CAddressApv1() << "-"<<(*it).getI2CAddressApv2()
         << " connected to Fed # " 
         << (*it).getFedNumber() << " Fed Channel # " 
	 << (*it).getFedChannel() << endl;
  }
}
// Check if it is the same module connection
bool ModuleConnection::isEqual(ModuleConnection mod){
  if (mod.getFecSlot() == this->getFecSlot() &&
      mod.getRingSlot() == this->getRingSlot() &&
      mod.getCcuAddress() == this->getCcuAddress() &&
      mod.getI2CChannel() == this->getI2CChannel() &&
      mod.getNumberOfPairs() == this->getNumberOfPairs())  return true;
  return false;    
}
