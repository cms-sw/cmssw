#ifndef DTUnpacker_h
#define DTUnpacker_h

/** \class DTUnpacker
 *  Base class for DT data unpackers
 *
 *  $Date: 2005/11/25 18:12:53 $
 *  $Revision: 1.2 $
 * \author M. Zanetti INFN Padova
 */


#include <FWCore/Framework/interface/ESHandle.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>

class DTReadOutMapping;

class DTUnpacker {

 public:
  
  /// Constructor
  DTUnpacker() {}

  /// Destructor
  virtual ~DTUnpacker() {}

  /// Unpacking method.
  /// index is the pointer to the beginning of the buffer.
  /// datasize is the size of the buffer in bytes
  virtual void interpretRawData(const unsigned int* index, int datasize,
				int dduID,
				edm::ESHandle<DTReadOutMapping>& mapping, 
				std::auto_ptr<DTDigiCollection>& product) = 0;

 protected:

};

#endif
