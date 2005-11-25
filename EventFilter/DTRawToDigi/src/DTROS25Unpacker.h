#ifndef DTROS25Unpacker_h
#define DTROS25Unpacker_h

/** \class DTROS25Unpacker
 *  The unpacker for DTs' ROS25: 
 *  final version of Read Out Sector board with 25 channels.
 *
 *  $Date: 2005/11/23 11:17:15 $
 *  $Revision: 1.4 $
 * \author M. Zanetti INFN Padova
 */

#include <EventFilter/DTRawToDigi/src/DTUnpacker.h>


class DTROS25Unpacker : public DTUnpacker {

 public:
  
  /// Constructor
  DTROS25Unpacker() {}

  /// Destructor
  virtual ~DTROS25Unpacker() {}

  /// Unpacking method
  virtual void interpretRawData(const unsigned int* index, int datasize,
				int dduID,
				edm::ESHandle<DTReadOutMapping>& mapping, 
				std::auto_ptr<DTDigiCollection>& product);

 private:


};

#endif
