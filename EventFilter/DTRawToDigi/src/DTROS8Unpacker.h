#ifndef DTROS8Unpacker_h
#define DTROS8Unpacker_h

/** \class DTROS8Unpacker
 *  The unpacker for DTs' ROS8: 
 *  final version of Read Out Sector board with 25 channels.
 *
 *  $Date: 2005/11/10 18:53:57 $
 *  $Revision: 1.1.2.1 $
 * \author M. Zanetti INFN Padova
 */


#include <FWCore/Framework/interface/ESHandle.h>

#include <CondFormats/DTMapping/interface/DTReadOutMapping.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>

using namespace edm;
using namespace std;

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
