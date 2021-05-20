#ifndef CSCFileReader_h
#define CSCFileReader_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "IORawData/DTCommissioning/plugins/RawFile.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include <vector>
#include <string>
#include <list>
#include <map>

#include "FileReaderDDU.h"
#include "FileReaderDCC.h"

class CSCFileReader : public edm::EDProducer {
private:
  std::vector<std::string> fileNames[40];
  std::vector<std::string>::const_iterator currentFile[40];

  int firstEvent, nEvents, tfDDUnumber;
  int expectedNextL1A, currentL1A[40];
  int nActiveRUIs, nActiveFUs;
  unsigned int runNumber;

  unsigned short *tmpBuf;
  const unsigned short *fuEvent[4];
  size_t fuEventSize[4];
  const unsigned short *ruBuf[40];
  size_t ruBufSize[40];

  FileReaderDDU RUI[40];
  FileReaderDCC FU[4];

  std::map<unsigned int, std::list<unsigned int> > FED;

  int readRUI(int rui, const unsigned short *&buf, size_t &length);
  int buildEventFromRUIs(FEDRawDataCollection *data);

  int readFU(int fu, const unsigned short *&buf, size_t &length);
  int nextEventFromFUs(FEDRawDataCollection *data);

public:
  CSCFileReader(const edm::ParameterSet &pset);
  ~CSCFileReader(void) override;

  virtual int fillRawData(edm::Event &e, /* edm::Timestamp& tstamp,*/ FEDRawDataCollection *&data);

  void produce(edm::Event &, edm::EventSetup const &) override;

  bool fFirstReadBug;
};

#endif
