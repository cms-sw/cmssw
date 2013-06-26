#ifndef DaqSource_DTROS25FileReader_h
#define DaqSource_DTROS25FileReader_h

/** \class DTROS25FileReader
 *  Read DT ROS8 raw data files
 *
 *  $Date: 2010/02/03 16:58:24 $
 *  $Revision: 1.6 $
 *  \author M. Zanetti - INFN Padova
 */

#include <IORawData/DaqSource/interface/DaqBaseReader.h>
#include <IORawData/DTCommissioning/src/RawFile.h>
#include "DataFormats/Provenance/interface/EventID.h"

#include <ostream>
#include <fstream>
#include <boost/cstdint.hpp>

class DTROS25FileReader : public DaqBaseReader {
 public:
  /// Constructor
  DTROS25FileReader(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTROS25FileReader();

  /// Generate and fill FED raw data for a full event
  virtual int fillRawData(edm::EventID& eID,
			  edm::Timestamp& tstamp, 
			  FEDRawDataCollection*& data);


  /// check for a 32 bits word to be a ROS25 header
  bool isHeader(uint32_t word);

  /// check for a 32 bits word to be a ROS25 trailer
  bool isTrailer(uint32_t word);

  /// swapping the lsBits with the msBits
  void swap(uint32_t & word);
 

  virtual bool checkEndOfFile();

 private:


  RawFile inputFile;

  edm::RunNumber_t runNumber;
  edm::EventNumber_t eventNumber;

  static const int rosWordLenght = 4;

};
#endif



