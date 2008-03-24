#ifndef EcalTBDaqRFIOFile_H
#define EcalTBDaqRFIOFile_H

#include "IORawData/EcalTBInputService/src/EcalTBDaqFile.h"

using namespace std;

class EcalTBDaqRFIOFile : public EcalTBDaqFile {

 public:

  /// Constructor
  EcalTBDaqRFIOFile(): filename_(), infile_(), isBinary_() {};

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
  FILE* infile_;
  bool isBinary_;

};
#endif
