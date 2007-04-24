#ifndef DTDDUUnpacker_h
#define DTDDUUnpacker_h

/** \class DTDDUUnpacker
 *  The unpacker for DTs' FED.
 *
 *  $Date: 2006/09/19 14:09:11 $
 *  $Revision: 1.10 $
 * \author M. Zanetti INFN Padova
 * FRC 060906
 */
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <EventFilter/DTRawToDigi/plugins/DTUnpacker.h>

class DTROS25Unpacker;
class DTDataMonitorInterface;

class DTDDUUnpacker : public DTUnpacker {

 public:
  
  /// Constructor
  DTDDUUnpacker(const edm::ParameterSet& ps);

  /// Destructor
  virtual ~DTDDUUnpacker();

  // Unpacking method
  virtual void interpretRawData(const unsigned int* index, int datasize,
				int dduID,
				edm::ESHandle<DTReadOutMapping>& mapping, 
				std::auto_ptr<DTDigiCollection>& product,
				std::auto_ptr<DTLocalTriggerCollection>& product2,
				uint16_t rosList=0);

 private:

  const edm::ParameterSet pset;

  bool debug;

  DTROS25Unpacker* ros25Unpacker;

  DTDataMonitorInterface * dataMonitor;

};

#endif
