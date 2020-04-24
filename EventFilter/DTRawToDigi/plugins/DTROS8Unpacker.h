#ifndef DTROS8Unpacker_h
#define DTROS8Unpacker_h

/** \class DTROS8Unpacker
 *  The unpacker for DTs' ROS8: 
 *  final version of Read Out Sector board with 25 channels.
 *
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
  ~DTROS8Unpacker() override {}

  // Unpacking method
  void interpretRawData(const unsigned int* index, int datasize,
				int dduID,
				edm::ESHandle<DTReadOutMapping>& mapping, 
				std::unique_ptr<DTDigiCollection>& product,
                                std::unique_ptr<DTLocalTriggerCollection>& product2,
				uint16_t rosList = 0) override;

private:

  const edm::ParameterSet pset;

};

#endif
