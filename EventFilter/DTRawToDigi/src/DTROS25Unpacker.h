#ifndef DTROS25Unpacker_h
#define DTROS25Unpacker_h

/** \class DTROS25Unpacker
 *  The unpacker for DTs' ROS25: 
 *  final version of Read Out Sector board with 25 channels.
 *
 *  $Date: 2007/04/26 18:53:06 $
 *  $Revision: 1.14 $
 * \author M. Zanetti INFN Padova
 * FRC 060906
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <EventFilter/DTRawToDigi/src/DTUnpacker.h>

class DTDataMonitorInterface;
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

  bool globalDAQ;

  const edm::ParameterSet pset;

  bool debug;
  bool writeSC;

  DTDataMonitorInterface * dataMonitor;

  std::vector<DTROS25Data> controlDataFromAllROS; 

};

#endif
