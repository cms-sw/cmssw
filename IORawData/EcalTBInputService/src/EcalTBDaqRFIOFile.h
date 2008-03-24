#ifndef EcalTBDaqRFIOFile_H
#define EcalTBDaqRFIOFile_H

#include "IORawData/EcalTBInputService/src/EcalTBDaqFile.h"

#include <string>
#include <stdio.h>

using namespace std;

class EcalTBDaqRFIOFile : public EcalTBDaqFile {

 public:

  /// Constructor
  EcalTBDaqRFIOFile(): filename_(), isBinary_(), infile_() {};

  /// Constructor
  EcalTBDaqRFIOFile(const std::string& filename, const bool& isBinary);

  /// Destructor
  virtual ~EcalTBDaqRFIOFile() 
    {
      close();
    };

  //Check if the position in file is EOF
  virtual bool checkEndOfFile();

  //Seek in the file for the event 
  virtual bool getEventData(FedDataPair& data);

  virtual void close();

 protected:

  std::string filename_;
  bool isBinary_;
  FILE* infile_;

};
#endif
