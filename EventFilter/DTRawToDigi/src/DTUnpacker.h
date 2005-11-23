#ifndef DTUnpacker_h
#define DTUnpacker_h

/** \class DTUnpacker
 *  Base class for DT data unpackers
 *
 *  $Date: 2005/11/22 13:52:15 $
 *  $Revision: 1.3 $
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

  /// Unpacking method
  virtual void interpretRawData(const unsigned char* index, int datasize,
				int dduID,
				edm::ESHandle<DTReadOutMapping>& mapping, 
				std::auto_ptr<DTDigiCollection>& product) = 0;

 protected:

};

#endif
