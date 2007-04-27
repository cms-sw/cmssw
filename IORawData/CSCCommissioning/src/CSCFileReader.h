#ifndef CSCFileReader_h
#define CSCFileReader_h

#include <IORawData/DaqSource/interface/DaqBaseReader.h>
#include <DataFormats/Provenance/interface/EventID.h>

#include <vector>
#include <string>
#include <list>
#include <map>

#include "FileReaderDDU.h"

class CSCFileReader : public DaqBaseReader {
private:
	std::vector<std::string> fileNames[40];
	std::vector<std::string>::const_iterator currentFile[40];

	int firstEvent, nEvents;
	int expectedNextL1A, currentL1A[40];

	unsigned short *tmpBuf;

	FileReaderDDU RUI[40];

	std::map<unsigned int,std::list<unsigned int> > FED;

	int readEvent(int rui, const unsigned short* &buf, size_t &length);
public:
	bool fillRawData(edm::EventID& eID, edm::Timestamp& tstamp, FEDRawDataCollection *& data);

	CSCFileReader(const edm::ParameterSet& pset);
	virtual ~CSCFileReader(void);
};

#endif
