#ifndef CSCFileReader_h
#define CSCFileReader_h

#include <IORawData/DaqSource/interface/DaqBaseReader.h>
#include <FWCore/EDProduct/interface/EventID.h>

class CSCFileReader : public DaqBaseReader {
private:
public:
	bool fillRawData(edm::EventID& eID, edm::Timestamp& tstamp, FEDRawDataCollection& data);

	CSCFileReader(const edm::ParameterSet& pset);
	virtual ~CSCFileReader(void){}
};

#endif
