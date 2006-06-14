#ifndef EcalTBDaqFile_H
#define EcalTBDaqFile_H

#include "IORawData/EcalTBInputService/src/EcalTBDaqFileReader.h"

using namespace std;


class EcalTBDaqFile  {

 public:

  /// Constructor
  EcalTBDaqFile() {};
  /// Destructor
  virtual ~EcalTBDaqFile() {};

  //Check if the position in file is EOF
  virtual bool checkEndOfFile() { return true; } 

  //Seek in the file for the event 
  virtual bool getEventData(FedDataPair& data) { return false; };

  virtual void close() {};
 protected:

  //static const int maxEventSizeInBytes_=42640;
  static const int maxEventSizeInBytes_=100000;
  static const int EOE_=10;
  static const int BOE_=5;
};
#endif
