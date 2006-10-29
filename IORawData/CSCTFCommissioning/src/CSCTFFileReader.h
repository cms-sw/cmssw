#ifndef CSCTFFileReader_h
#define CSCTFFileReader_h

#include <IORawData/DaqSource/interface/DaqBaseReader.h>
#include <DataFormats/Common/interface/EventID.h>

class SPReader;

class CSCTFFileReader : public DaqBaseReader 
{
 private:
  SPReader* ___ddu;
  
 public:
  bool fillRawData(edm::EventID& eID, edm::Timestamp& tstamp, FEDRawDataCollection*& data);
  
  CSCTFFileReader(const edm::ParameterSet& pset);
  virtual ~CSCTFFileReader();
};

#endif
