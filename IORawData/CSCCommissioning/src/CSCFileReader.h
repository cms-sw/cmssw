#ifndef CSCFileReader_h
#define CSCFileReader_h

#include <IORawData/DaqSource/interface/DaqBaseReader.h>
#include <DataFormats/Provenance/interface/EventID.h>

#include <vector>
#include <string>
#include <list>
#include <map>

#include "FileReaderDDU.h"
#include "FileReaderDCC.h"

class CSCFileReader : public DaqBaseReader {
private:
	std::vector<std::string> fileNames[40];
	std::vector<std::string>::const_iterator currentFile[40];

	int firstEvent, nEvents, tfDDUnumber;
	int expectedNextL1A, currentL1A[40];
	int nActiveRUIs, nActiveFUs;

	unsigned short *tmpBuf;
	const unsigned short *fuEvent[4];
	size_t fuEventSize[4];
	const unsigned short *ruBuf[40];
	size_t ruBufSize[40];

	FileReaderDDU RUI[40];
	FileReaderDCC FU [4];

	std::map<unsigned int,std::list<unsigned int> > FED;

	int readRUI(int rui, const unsigned short* &buf, size_t &length);
	int buildEventFromRUIs(FEDRawDataCollection *data);

	int readFU (int fu,  const unsigned short* &buf, size_t &length);
	int nextEventFromFUs  (FEDRawDataCollection *data);

public:
	int fillRawData(edm::EventID& eID, edm::Timestamp& tstamp, FEDRawDataCollection *& data);

	CSCFileReader(const edm::ParameterSet& pset);
	virtual ~CSCFileReader(void);
};

#endif
