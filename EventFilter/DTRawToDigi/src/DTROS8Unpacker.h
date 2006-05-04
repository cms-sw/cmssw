#ifndef DTROS8Unpacker_h
#define DTROS8Unpacker_h

/** \class DTROS8Unpacker
 *  The unpacker for DTs' ROS8: 
 *  final version of Read Out Sector board with 25 channels.
 *
 *  $Date: 2005/11/25 18:12:53 $
 *  $Revision: 1.5 $
 * \author M. Zanetti INFN Padova
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <EventFilter/DTRawToDigi/src/DTUnpacker.h>

class DTReadOutMapping;

class DTROS8Unpacker : public DTUnpacker {

public:
  
  /// Constructor
  DTROS8Unpacker(const edm::ParameterSet& ps): pset(ps) {}

  /// Destructor
  virtual ~DTROS8Unpacker() {}

  /// Unpacking method
  virtual void interpretRawData(const unsigned int* index, int datasize,
				int dduID,
				edm::ESHandle<DTReadOutMapping>& mapping, 
				std::auto_ptr<DTDigiCollection>& product);

private:

  const edm::ParameterSet pset;

};

#endif
