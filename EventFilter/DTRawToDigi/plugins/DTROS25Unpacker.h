#ifndef DTROS25Unpacker_h
#define DTROS25Unpacker_h

/** \class DTROS25Unpacker
 *  The unpacker for DTs' ROS25: 
 *  final version of Read Out Sector board with 25 channels.
 *
 * \author M. Zanetti INFN Padova
 * FRC 060906
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <EventFilter/DTRawToDigi/plugins/DTUnpacker.h>

class DTROS25Data;

class DTROS25Unpacker : public DTUnpacker {

public:
  
  /// Constructor
  DTROS25Unpacker(const edm::ParameterSet& ps);

  /// Destructor
  virtual ~DTROS25Unpacker();

  // Unpacking method
  virtual void interpretRawData(const unsigned int* index, int datasize,
				int dduID,
				edm::ESHandle<DTReadOutMapping>& mapping, 
				std::auto_ptr<DTDigiCollection>& product,
				std::auto_ptr<DTLocalTriggerCollection>& product2,
				uint16_t rosList = 0);

  inline const std::vector<DTROS25Data> & getROSsControlData() const {
    return controlDataFromAllROS;
  }

private:

  int swap(int x);

  /// if reading data locally, words, being assembled as 32-bits, do not need to be swapped
  bool localDAQ;

  /// if data are read from ROS directly, no information on the ROS Id is present
  bool readingDDU;

  /// since June 2007, local DAQ, provides FED number
  bool readDDUIDfromDDU;
  /// to analyze older data..
  int hardcodedDDUID;

  /// make the local SC spy data persistent
  bool writeSC;

  /// perform DQM on ROS data
  bool performDataIntegrityMonitor;

  bool debug;

  std::vector<DTROS25Data> controlDataFromAllROS; 

};

#endif
