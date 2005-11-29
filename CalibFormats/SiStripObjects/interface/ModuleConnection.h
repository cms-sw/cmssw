#ifndef CALIBTRACKER_SISTRIPCONNECTIVITY_MODULECONNECTION_H
#define CALIBTRACKER_SISTRIPCONNECTIVITY_MODULECONNECTION_H

#include "CalibTracker/SiStripConnectivity/interface/ApvPairConnection.h"
#include<vector>
#include<set>
#include<string>
using namespace std;

class ModuleConnection {
 public:
  typedef vector<ApvPairConnection> VectorType;
  //
  // Construct from N ApvPairConnection
  //
  ModuleConnection(VectorType);
  //
  // Construct from parameters
  //
  //
  // Empty constructor
  //
  ModuleConnection();
  //
  // 
  //
  int getFecSlot() {return (((getApvPairs()).front()).getFecSlot());}
  int getRingSlot() {return (((getApvPairs()).front()).getRingSlot());}
  //
  // get methods
  //
  VectorType getApvPairs()  {consistencyChecks(); return apvPairs;}
  //
  // set methods
  //
  void setApvPairs(VectorType a);
  //
  // Add Methods
  //
  void addApvPair(ApvPairConnection);
  //
  // get methods
  //
  int getCcuAddress();
  int getI2CChannel();
  int getNumberOfPairs() {consistencyChecks(); return apvPairs.size();}
  //
  // Module ID
  //
  void setModuleId(string i){moduleId =i;}
  string getModuleId(){return moduleId;}
  //
  // DCU ID
  //
  void setDcuId(string i){dcuId =i;}
  string getDcuId(){return dcuId;}
  //
  // PART ID (barcode Id for a module)
  //
  void setPartId(string i){partId =i;}
  string getPartId(){return partId;}
  // Print 
  void print();

  bool consistencyChecks();

  //
  //
  //
  set<int> getConnectedFeds()  ;

  bool isEqual(ModuleConnection mod);

 private:
   
  VectorType apvPairs;
  
  string moduleId;
  string dcuId;
  string partId;

};


class ApvI2CAddressLess {
public:
  bool operator()( const ApvPairConnection& a, const ApvPairConnection&
b) {
    return a.getI2CAddressApv1() < b.getI2CAddressApv1();
  }
};



#endif

