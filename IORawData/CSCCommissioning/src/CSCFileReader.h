#ifndef CSCFileReader_h
#define CSCFileReader_h

#include <IORawData/DaqSource/interface/DaqBaseReader.h>
#include <DataFormats/Common/interface/EventID.h>
#include "FileReaderDDU.h"

class CSCFileReader : public DaqBaseReader {
private:
  FileReaderDDU ___ddu;
public:
	bool fillRawData(edm::EventID& eID, edm::Timestamp& tstamp, FEDRawDataCollection& data);

	CSCFileReader(const edm::ParameterSet& pset);
	virtual ~CSCFileReader(void){}
};

#endif
