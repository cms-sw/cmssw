#ifndef DaqSource_DTROS8FileReader_h
#define DaqSource_DTROS8FileReader_h

/** \class DTROS8FileReader
 *  Read DT ROS8 raw data files
 *
 *  $Date: 2007/03/12 01:01:57 $
 *  $Revision: 1.7 $
 *  \author M. Zanetti - INFN Padova
 */

#include <IORawData/DaqSource/interface/DaqBaseReader.h>
#include <IORawData/DTCommissioning/src/RawFile.h>
#include "DataFormats/Provenance/interface/EventID.h"

#include <fstream>

class DTROS8FileReader : public DaqBaseReader {
 public:
  /// Constructor
  DTROS8FileReader(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTROS8FileReader();

  /// Generate and fill FED raw data for a full event
  virtual int fillRawData(edm::EventID& eID,
			  edm::Timestamp& tstamp, 
			  FEDRawDataCollection*& data);

  virtual bool checkEndOfFile();

 private:

  RawFile inputFile;

  edm::RunNumber_t runNum;
  edm::EventNumber_t eventNum;

  static const int ros8WordLenght = 4;

};
#endif

