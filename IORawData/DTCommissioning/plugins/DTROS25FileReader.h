#ifndef DaqSource_DTROS25FileReader_h
#define DaqSource_DTROS25FileReader_h

/** \class DTROS25FileReader
 *  Read DT ROS8 raw data files
 *
 *  $Date: 2010/02/03 16:58:24 $
 *  $Revision: 1.6 $
 *  \author M. Zanetti - INFN Padova
 */

#include "FWCore/Framework/interface/one/EDProducer.h"
#include "IORawData/DTCommissioning/plugins/RawFile.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include <ostream>
#include <fstream>
#include <cstdint>

class DTROS25FileReader : public edm::one::EDProducer<> {
public:
  /// Constructor
  DTROS25FileReader(const edm::ParameterSet& pset);

  /// Destructor
  ~DTROS25FileReader() override;

  /// Generate and fill FED raw data for a full event
  virtual int fillRawData(edm::Event& e,
                          //			  edm::Timestamp& tstamp,
                          FEDRawDataCollection*& data);

  void produce(edm::Event&, edm::EventSetup const&) override;

  /// check for a 32 bits word to be a ROS25 header
  bool isHeader(uint32_t word);

  /// check for a 32 bits word to be a ROS25 trailer
  bool isTrailer(uint32_t word);

  /// swapping the lsBits with the msBits
  void swap(uint32_t& word);

  virtual bool checkEndOfFile();

private:
  RawFile inputFile;

  edm::RunNumber_t runNumber;
  edm::EventNumber_t eventNumber;

  static const int rosWordLenght = 4;
};
#endif
