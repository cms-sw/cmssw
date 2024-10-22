#ifndef DaqSource_DTSpyReader_h
#define DaqSource_DTSpyReader_h

/** \class DTSpyReader
 *  Read DT ROS8 raw data files
 *
 *  $Date: 2010/02/03 16:58:24 $
 *  $Revision: 1.4 $
 *  \author M. Zanetti - INFN Padova
 */
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "IORawData/DTCommissioning/plugins/RawFile.h"
#include "IORawData/DTCommissioning/plugins/DTSpy.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include <ostream>
#include <fstream>
#include <cstdint>

class DTSpyReader : public edm::one::EDProducer<> {
public:
  /// Constructor
  DTSpyReader(const edm::ParameterSet& pset);

  /// Destructor
  ~DTSpyReader() override;

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
  uint64_t dmaUnpack(const uint32_t* dmaData, bool& isData);

  /// swapping the lsBits with the msBits
  void swap(uint64_t& word);

private:
  DTSpy* mySpy;

  edm::RunNumber_t runNumber;
  edm::EventNumber_t eventNumber;

  bool debug;
  int dduID;

  static const int dduWordLength = 8;
};
#endif
