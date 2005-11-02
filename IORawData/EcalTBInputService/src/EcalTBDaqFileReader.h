#ifndef EcalTBDaqFileReader_H
#define EcalTBDaqFileReader_H

#include "IORawData/EcalTBInputService/src/EcalTBDaqFileReader.h"

#include <FWCore/EDProduct/interface/EventID.h>
#include <iosfwd>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;

class DaqInputEvent;
namespace edm {class EventID; class Timestamp;}
//namespace raw {class FEDRawDataCollection; }

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
  virtual void initialize(const std::string & filename, bool isBinary);
  virtual bool fillDaqEventData(edm::EventID & cID, FEDRawDataCollection& data);
  virtual FedDataPair getEventTrailer();
  virtual bool checkEndOfEvent();
  virtual bool checkEndOfFile();
  void getFEDHeader(unsigned long* buf);
  int getFedId() {return headValues_[0];}
  int getEventNumber() {return headValues_[1];}
  int getEventLength() {return headValues_[2];}
  int getRunNumber() {return headValues_[3];}

  

private:
  static EcalTBDaqFileReader * instance_;
  ifstream inputFile;

  //static const int maxEventSizeInBytes_=42640;
  static const int maxEventSizeInBytes_=100000;
  static const int EOE_=10;
  static const int BOE_=5;
  vector<int> headValues_;
  
  //ulong* buf;
  //int len;
  //ulong* tmp;
  //unsigned char* fedData;

protected:

  bool initialized_;
  bool isBinary_;
  //std::ifstream * input_;
};
#endif
