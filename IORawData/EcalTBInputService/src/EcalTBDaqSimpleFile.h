#ifndef EcalTBDaqSimpleFile_H
#define EcalTBDaqSimpleFile_H

#include "IORawData/EcalTBInputService/src/EcalTBDaqFile.h"

#include <string>
#include <fstream>

using namespace std;

class EcalTBDaqSimpleFile : public EcalTBDaqFile  {

 public:

  /// Constructor
  EcalTBDaqSimpleFile(): filename_(), isBinary_(0), infile_() {};

  /// Constructor
  EcalTBDaqSimpleFile(const std::string& filename, const bool& isBinary);

  /// Destructor
  virtual ~EcalTBDaqSimpleFile() 
    {
      close();
    };

  //Check if the position in file is EOF
  virtual bool checkEndOfFile() ;

  //Seek in the file for the event 
  virtual bool getEventData(FedDataPair& data);

  virtual void close();

 protected:

  std::string filename_;
  bool isBinary_;
  ifstream infile_;

};
#endif
