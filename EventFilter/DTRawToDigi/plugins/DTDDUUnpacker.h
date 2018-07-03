#ifndef DTDDUUnpacker_h
#define DTDDUUnpacker_h

/** \class DTDDUUnpacker
 *  The unpacker for DTs' FED.
 *
 * \author M. Zanetti INFN Padova
 * FRC 060906
 */
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <EventFilter/DTRawToDigi/plugins/DTUnpacker.h>
#include <EventFilter/DTRawToDigi/plugins/DTROS25Unpacker.h>
#include "DataFormats/DTDigi/interface/DTControlData.h"
// class DTROS25Unpacker;

class DTDDUUnpacker : public DTUnpacker {

 public:
  
  /// Constructor
  DTDDUUnpacker(const edm::ParameterSet& ps);

  /// Destructor
  ~DTDDUUnpacker() override;

  // Unpacking method
  void interpretRawData(const unsigned int* index, int datasize,
				int dduID,
				edm::ESHandle<DTReadOutMapping>& mapping, 
				std::unique_ptr<DTDigiCollection>& product,
				std::unique_ptr<DTLocalTriggerCollection>& product2,
				uint16_t rosList=0) override;
  
  inline const std::vector<DTROS25Data> & getROSsControlData() const {
    return ros25Unpacker->getROSsControlData();
  }
  
  inline const DTDDUData & getDDUControlData() const {
    return controlData;
  }
  
 private:

  const edm::ParameterSet dduPSet;

  /// if data are read locally, status words are swapped
  bool localDAQ;
  
  /// perform DQM for DDU
  bool performDataIntegrityMonitor;

  bool debug;

  DTROS25Unpacker* ros25Unpacker;

  DTDDUData controlData;
  
};

#endif
