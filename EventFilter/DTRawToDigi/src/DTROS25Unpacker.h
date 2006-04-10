#ifndef DTROS25Unpacker_h
#define DTROS25Unpacker_h

/** \class DTROS25Unpacker
 *  The unpacker for DTs' ROS25: 
 *  final version of Read Out Sector board with 25 channels.
 *
 *  $Date: 2006/04/07 15:36:04 $
 *  $Revision: 1.7 $
 * \author M. Zanetti INFN Padova
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <EventFilter/DTRawToDigi/src/DTUnpacker.h>

class DTDataMonitorInterface;

class DTROS25Unpacker : public DTUnpacker {

public:
  
  /// Constructor
  DTROS25Unpacker(const edm::ParameterSet& ps);

  /// Destructor
  virtual ~DTROS25Unpacker() {}

  // Unpacking method
  virtual void interpretRawData(const unsigned int* index, int datasize,
				int dduID,
				edm::ESHandle<DTReadOutMapping>& mapping, 
				std::auto_ptr<DTDigiCollection>& product,
				uint16_t rosList = 0);

private:

  const edm::ParameterSet pset;

  DTDataMonitorInterface * dataMonitor;

};

#endif
