#ifndef DTDDUUnpacker_h
#define DTDDUUnpacker_h

/** \class DTDDUUnpacker
 *  The unpacker for DTs' FED.
 *
 *  $Date: 2005/11/21 17:38:48 $
 *  $Revision: 1.2 $
 * \author M. Zanetti INFN Padova
 */

#include <EventFilter/DTRawToDigi/src/DTUnpacker.h>

class DTROS25Unpacker;

class DTDDUUnpacker : public DTUnpacker {

 public:
  
  /// Constructor
  DTDDUUnpacker();

  /// Destructor
  virtual ~DTDDUUnpacker();

  /// Unpacking method
  virtual void interpretRawData(const unsigned char* index, int datasize,
				int dduID,
				edm::ESHandle<DTReadOutMapping>& mapping, 
				std::auto_ptr<DTDigiCollection>& product);

 private:
  DTROS25Unpacker* ros25Unpacker;

};

#endif
