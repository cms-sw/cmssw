#ifndef EcalTBDaqFileReader_H
#define EcalTBDaqFileReader_H

#include "IORawData/EcalTBInputService/src/EcalTBDaqFileReader.h"

#include <FWCore/EDProduct/interface/CollisionID.h>
#include <iosfwd>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;

class DaqInputEvent;
class DaqFEDRawData;
namespace raw {class FEDRawDataCollection; }

struct FedDataPair {
  unsigned char* fedData;
  int len;
};

class EcalTBDaqFileReader  {

 public:

  /// Constructor
  EcalTBDaqFileReader();


  /// Destructor
  virtual ~EcalTBDaqFileReader();


  static EcalTBDaqFileReader * instance();
  

  void setInitialized(bool value);
  bool isInitialized();

  // Override virtual methods from DaqFileReader
  virtual void initialize(const std::string & filename);
  virtual bool fillDaqEventData(edm::CollisionID & cID, raw::FEDRawDataCollection& data);
  virtual FedDataPair getEventTrailer();
  virtual bool checkEndOfEvent();
  virtual bool checkEndOfFile();

private:
  static EcalTBDaqFileReader * instance_;
  ifstream inputFile;

  static const int maxEventSizeInBytes_=41544;
  static const int EOE_=10;
  static const int BOE_=5;
  //ulong* buf;
  //int len;
  //ulong* tmp;
  //unsigned char* fedData;

protected:

  bool initialized_;
  //std::ifstream * input_;
};
#endif
