#ifndef DaqSource_DTDDUFileReader_h
#define DaqSource_DTDDUFileReader_h

/** \class DTDDUFileReader
 *  Read DT ROS8 raw data files
 *
 *  $Date: 2010/02/03 16:58:24 $
 *  $Revision: 1.11 $
 *  \author M. Zanetti - INFN Padova
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "IORawData/DTCommissioning/plugins/RawFile.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include <ostream>
#include <fstream>
#include <cstdint>

class DTDDUFileReader : public edm::EDProducer {
public:
  /// Constructor
  DTDDUFileReader(const edm::ParameterSet& pset);

  /// Destructor
  ~DTDDUFileReader() override;

  /// Generate and fill FED raw data for a full event
  virtual int fillRawData(edm::Event& e,
                          //			  edm::Timestamp& tstamp,
                          FEDRawDataCollection*& data);

  void produce(edm::Event&, edm::EventSetup const&) override;

  /// check for a 64 bits word to be a DDU header
  bool isHeader(uint64_t word, bool dataTag);

  /// check for a 64 bits word to be a DDU trailer
  bool isTrailer(uint64_t word, bool dataTag, unsigned int wordCount);

  /// pre-unpack the data if read via DMA
  //  std::pair<uint64_t,bool> dmaUnpack();
  uint64_t dmaUnpack(bool& isData, int& nread);

  /// swapping the lsBits with the msBits
  void swap(uint64_t& word);

  virtual bool checkEndOfFile();

private:
  RawFile inputFile;

  edm::RunNumber_t runNumber;
  edm::EventNumber_t eventNumber;

  int dduID;

  bool readFromDMA;
  int skipEvents;
  int numberOfHeaderWords;

  static const int dduWordLength = 8;
};
#endif
