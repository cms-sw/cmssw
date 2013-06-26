#ifndef DaqSource_DTSpyReader_h
#define DaqSource_DTSpyReader_h

/** \class DTSpyReader
 *  Read DT ROS8 raw data files
 *
 *  $Date: 2010/02/03 16:58:24 $
 *  $Revision: 1.4 $
 *  \author M. Zanetti - INFN Padova
 */

#include <IORawData/DaqSource/interface/DaqBaseReader.h>
#include <IORawData/DTCommissioning/src/RawFile.h>
#include "IORawData/DTCommissioning/src/DTSpy.h"

#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include "DataFormats/Provenance/interface/EventID.h"

#include <ostream>
#include <fstream>
#include <boost/cstdint.hpp>

class DTSpyReader : public DaqBaseReader {
 public:
  /// Constructor
  DTSpyReader(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTSpyReader();

  /// Generate and fill FED raw data for a full event
  virtual int fillRawData(edm::EventID& eID,
			  edm::Timestamp& tstamp, 
			  FEDRawDataCollection*& data);


  /// check for a 64 bits word to be a DDU header
  bool isHeader(uint64_t word, bool dataTag);

  /// check for a 64 bits word to be a DDU trailer
  bool isTrailer(uint64_t word, bool dataTag, int wordCount);

  /// pre-unpack the data if read via DMA
  //  std::pair<uint64_t,bool> dmaUnpack();
  uint64_t dmaUnpack(const uint32_t *dmaData ,bool & isData);

  /// swapping the lsBits with the msBits
  void swap(uint64_t & word);
 

 private:

  DTSpy * mySpy;

  edm::RunNumber_t runNumber;
  edm::EventNumber_t eventNumber;

  bool debug;
  int dduID; 

  static const int dduWordLength = 8;

};
#endif



