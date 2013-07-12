#ifndef DTUnpacker_h
#define DTUnpacker_h

/** \class DTUnpacker
 *  Base class for DT data unpackers
 *
 *  $Date: 2007/04/24 12:08:20 $
 *  $Revision: 1.1 $
 * \author M. Zanetti INFN Padova
 *  FR 060906
 */


#include <FWCore/Framework/interface/ESHandle.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>
#include <DataFormats/DTDigi/interface/DTLocalTriggerCollection.h>

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
				std::auto_ptr<DTDigiCollection>& product, 
				std::auto_ptr<DTLocalTriggerCollection>& product2, 
                                uint16_t rosList=0) = 0;

 protected:

};

#endif
