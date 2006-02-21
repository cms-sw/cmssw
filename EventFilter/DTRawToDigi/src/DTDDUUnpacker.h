#ifndef DTDDUUnpacker_h
#define DTDDUUnpacker_h

/** \class DTDDUUnpacker
 *  The unpacker for DTs' FED.
 *
 *  $Date: 2005/11/25 18:12:53 $
 *  $Revision: 1.4 $
 * \author M. Zanetti INFN Padova
 */
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <EventFilter/DTRawToDigi/src/DTUnpacker.h>

class DTROS25Unpacker;

class DTDDUUnpacker : public DTUnpacker {

 public:
  
  /// Constructor
  DTDDUUnpacker(const edm::ParameterSet& ps);

  /// Destructor
  virtual ~DTDDUUnpacker();

  /// Unpacking method
  virtual void interpretRawData(const unsigned int* index, int datasize,
				int dduID,
				edm::ESHandle<DTReadOutMapping>& mapping, 
				std::auto_ptr<DTDigiCollection>& product);

 private:

  const edm::ParameterSet pset;

  DTROS25Unpacker* ros25Unpacker;


};

#endif
