#ifndef DTROS25Unpacker_h
#define DTROS25Unpacker_h

/** \class DTROS25Unpacker
 *  The unpacker for DTs' ROS25: 
 *  final version of Read Out Sector board with 25 channels.
 *
 *  $Date: 2005/11/22 14:16:43 $
 *  $Revision: 1.3 $
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
  virtual void interpretRawData(const unsigned char* index, int datasize,
				int dduID,
				edm::ESHandle<DTReadOutMapping>& mapping, 
				std::auto_ptr<DTDigiCollection>& product);

 private:


};

#endif
