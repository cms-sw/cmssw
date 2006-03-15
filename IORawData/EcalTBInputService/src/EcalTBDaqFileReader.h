#ifndef EcalTBDaqFileReader_H
#define EcalTBDaqFileReader_H

#include "IORawData/EcalTBInputService/src/EcalTBDaqFileReader.h"

#include <DataFormats/Common/interface/EventID.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iosfwd>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;

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

  //Return the instance of the reader
  static EcalTBDaqFileReader * instance();

  //Set the initialization bit
  void setInitialized(bool value);
  //Check if initialized or not
  bool isInitialized();
  //Initialization 
  void initialize(const std::string & filename, bool isBinary);

  //Check if the position in file is EOF
  bool checkEndOfFile();

  //Fill Data for an event from input file
  bool fillDaqEventData();

  //Return cachedData for the event 
  const FedDataPair& getFedData() { return cachedData_;} 
  //Return event FedId
  int getFedId() {return headValues_[0];}
  //Return event number
  int getEventNumber() {return headValues_[1];}
  //Return event size
  int getEventLength() {return headValues_[2];}
  //Return run number
  int getRunNumber() {return headValues_[3];}

protected:

  bool initialized_;
  bool isBinary_;
  
private:

  //Fill event header information
  void setFEDHeader();
  //Seek in the file for the event 
  void getEventTrailer();

  ifstream inputFile;

  static EcalTBDaqFileReader * instance_;

  //static const int maxEventSizeInBytes_=42640;
  static const int maxEventSizeInBytes_=100000;
  static const int EOE_=10;
  static const int BOE_=5;

  vector<int> headValues_;
  FedDataPair cachedData_;



};
#endif
