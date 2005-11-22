#ifndef DTROS25Unpacker_h
#define DTROS25Unpacker_h

/** \class DTROS25Unpacker
 *  The unpacker for DTs' ROS25: 
 *  final version of Read Out Sector board with 25 channels.
 *
 *  $Date: 2005/11/21 17:38:48 $
 *  $Revision: 1.2 $
 * \author M. Zanetti INFN Padova
 */


#include <FWCore/Framework/interface/ESHandle.h>

#include <DataFormats/DTDigi/interface/DTDigiCollection.h>

class DTReadOutMapping;

class DTROS25Unpacker {

 public:
  
  /// Constructor
  DTROS25Unpacker() {}

  /// Destructor
  virtual ~DTROS25Unpacker() {}

  /// Unpacking method
  void interpretRawData(const unsigned char* index, int datasize,
			int dduID,
			edm::ESHandle<DTReadOutMapping>& mapping, 
			std::auto_ptr<DTDigiCollection>& product);

 private:


};

#endif
