#ifndef CSCTFFileReader_h
#define CSCTFFileReader_h

#include <IORawData/DaqSource/interface/DaqBaseReader.h>
#include <FWCore/EDProduct/interface/EventID.h>

class CSCTFFileReader : public DaqBaseReader {
private:
public:
	bool fillRawData(edm::EventID& eID, edm::Timestamp& tstamp, FEDRawDataCollection& data);

	CSCTFFileReader(const edm::ParameterSet& pset);
	virtual ~CSCTFFileReader(void){}
};

#endif
