#ifndef DaqSource_DTROS8FileReader_h
#define DaqSource_DTROS8FileReader_h

/** \class DTROS8FileReader
 *  Read DT ROS8 raw data files
 *
 *  $Date: 2010/02/03 16:58:24 $
 *  $Revision: 1.8 $
 *  \author M. Zanetti - INFN Padova
 */

#include "FWCore/Framework/interface/one/EDProducer.h"
#include "IORawData/DTCommissioning/plugins/RawFile.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include <fstream>

class DTROS8FileReader : public edm::one::EDProducer<> {
public:
  /// Constructor
  DTROS8FileReader(const edm::ParameterSet& pset);

  /// Destructor
  ~DTROS8FileReader() override;

  /// Generate and fill FED raw data for a full event
  virtual int fillRawData(edm::Event& e,
                          //			  edm::Timestamp& tstamp,
                          FEDRawDataCollection*& data);

  void produce(edm::Event&, edm::EventSetup const&) override;

  virtual bool checkEndOfFile();

private:
  RawFile inputFile;

  edm::RunNumber_t runNum;
  edm::EventNumber_t eventNum;

  static const int ros8WordLenght = 4;
};
#endif
