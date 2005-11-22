#ifndef DTROS8Unpacker_h
#define DTROS8Unpacker_h

/** \class DTROS8Unpacker
 *  The unpacker for DTs' ROS8: 
 *  final version of Read Out Sector board with 25 channels.
 *
 *  $Date: 2005/11/21 17:38:48 $
 *  $Revision: 1.2 $
 * \author M. Zanetti INFN Padova
 */


#include <FWCore/Framework/interface/ESHandle.h>

#include <DataFormats/DTDigi/interface/DTDigiCollection.h>

class DTReadOutMapping;

class DTROS8Unpacker {

 public:
  
  /// Constructor
  DTROS8Unpacker() {}

  /// Destructor
  virtual ~DTROS8Unpacker() {}

  /// Unpacking method
  void interpretRawData(const unsigned char* index, int datasize,
			edm::ESHandle<DTReadOutMapping>& mapping, 
			std::auto_ptr<DTDigiCollection>& product);

 private:


};

#endif
