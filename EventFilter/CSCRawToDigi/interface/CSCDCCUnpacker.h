#ifndef CSCDCCUnpacker_h
#define CSCDCCUnpacker_h

/** \class CSCDCCUnpacker
 * 
 *
 *  $Date: 2008/06/24 15:51:49 $
 *  $Revision: 1.17 $
 * \author Alex Tumanov 
 */

#include <FWCore/Framework/interface/EDProducer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include "FWCore/ParameterSet/interface/InputTag.h"

class CSCMonitorInterface;

class CSCDCCUnpacker: public edm::EDProducer {
 public:
  /// Constructor
  CSCDCCUnpacker(const edm::ParameterSet & pset);
  
  /// Destructor
  virtual ~CSCDCCUnpacker();
  
  /// Produce digis out of raw data
  void produce(edm::Event & e, const edm::EventSetup& c);



  
 private:

  bool debug, printEventNumber, goodEvent, useExaminer, unpackStatusDigis, unpackMTCCData;
  bool useSelectiveUnpacking, useFormatStatus; 
  int numOfEvents;
  unsigned int errorMask, examinerMask;
  bool instatiateDQM;
  CSCMonitorInterface * monitor;
  edm::InputTag inputObjectsTag; // input tag labelling raw data for input



};

#endif
