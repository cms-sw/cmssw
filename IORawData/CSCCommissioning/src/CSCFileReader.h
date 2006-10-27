#ifndef CSCFileReader_h
#define CSCFileReader_h

#include <IORawData/DaqSource/interface/DaqBaseReader.h>
#include <DataFormats/Common/interface/EventID.h>

#include <vector>
#include <string>

class CSCFileReader : public DaqBaseReader {
private:
	std::vector<std::string> fileNames;
	std::vector<std::string>::const_iterator currentFile;
	enum {DAQ=1,TF=2};
	int  dataType;

public:
	bool fillRawData(edm::EventID& eID, edm::Timestamp& tstamp, FEDRawDataCollection *& data);

	CSCFileReader(const edm::ParameterSet& pset);
	virtual ~CSCFileReader(void){}
};

#endif
