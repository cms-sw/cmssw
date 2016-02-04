#ifndef DTROS8Unpacker_h
#define DTROS8Unpacker_h

/** \class DTROS8Unpacker
 *  The unpacker for DTs' ROS8: 
 *  final version of Read Out Sector board with 25 channels.
 *
 *  $Date: 2007/05/07 16:16:39 $
 *  $Revision: 1.3 $
 * \author M. Zanetti INFN Padova
 *  FRC 140906 
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <EventFilter/DTRawToDigi/plugins/DTUnpacker.h>

class DTReadOutMapping;

class DTROS8Unpacker : public DTUnpacker {

public:
  
  /// Constructor
  DTROS8Unpacker(const edm::ParameterSet& ps): pset(ps) {}

  /// Destructor
  virtual ~DTROS8Unpacker() {}

  // Unpacking method
  virtual void interpretRawData(const unsigned int* index, int datasize,
				int dduID,
				edm::ESHandle<DTReadOutMapping>& mapping, 
				std::auto_ptr<DTDigiCollection>& product,
                                std::auto_ptr<DTLocalTriggerCollection>& product2,
				uint16_t rosList = 0);

private:

  const edm::ParameterSet pset;

};

#endif
