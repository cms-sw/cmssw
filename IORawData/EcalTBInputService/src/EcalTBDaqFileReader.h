#ifndef EcalTBDaqFileReader_H
#define EcalTBDaqFileReader_H




#include <iosfwd>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

//namespace raw {class FEDRawDataCollection; }

//forward declaration

class EcalTBDaqFile;

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

  //Set the initialization bit
  void setInitialized(bool value);
  //Check if initialized or not
  bool isInitialized();

  //Initialization 
  void initialize(const std::string & filename, bool isBinary);

  //Fill Data for an event from input file
  bool fillDaqEventData();

  //Return the DAQ File
  const EcalTBDaqFile* getDaqFile() const { return inputFile_; }

  //Check if the position in file is EOF
  bool checkEndOfFile() const;  

  //Return cachedData for the event 
  const FedDataPair& getFedData() const { return cachedData_;} 

  //Return event FedId
  int getFedId() const {return headValues_[0];} 

  //Return event number
  int getEventNumber() const {return headValues_[1];}

  //Return event size
  int getEventLength() const {return headValues_[2];}

  //Return run number
  int getRunNumber() const {return headValues_[3];}

protected:

  bool initialized_;
  bool isBinary_;
  
private:

  //Fill event header information
  void setFEDHeader();

  EcalTBDaqFile* inputFile_;

  vector<int> headValues_;
  FedDataPair cachedData_;

};
#endif
