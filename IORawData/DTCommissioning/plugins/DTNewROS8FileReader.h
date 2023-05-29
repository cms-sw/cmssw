#ifndef DaqSource_DTNewROS8FileReader_h
#define DaqSource_DTNewROS8FileReader_h

/** \class DTNewROS8FileReader
 *  Read DT ROS8 raw data files
 *  From DTROS8FileReader
 *
 *  $Date: 2015/12/17$
 */

#include "FWCore/Framework/interface/one/EDProducer.h"
#include "IORawData/DTCommissioning/plugins/RawFile.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include <fstream>

class DTNewROS8FileReader : public edm::one::EDProducer<> {
public:
  /// Constructor
  DTNewROS8FileReader(const edm::ParameterSet& pset);

  /// Destructor
  ~DTNewROS8FileReader() override;

  /// Generate and fill FED raw data for a full event
  virtual int fillRawData(edm::Event& e,
                          //			  edm::Timestamp& tstamp,
                          FEDRawDataCollection*& data);

  void produce(edm::Event&, edm::EventSetup const&) override;

  virtual bool checkEndOfFile();

private:
  RawFile inputFile;

  edm::RunNumber_t runNumber;
  edm::EventNumber_t eventNum;

  static const int ros8WordLenght = 4;
};
#endif
