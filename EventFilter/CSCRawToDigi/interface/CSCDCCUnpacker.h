#ifndef CSCDCCUnpacker_h
#define CSCDCCUnpacker_h

/** \class CSCDCCUnpacker
 * 
 *
 *  $Date: 2007/04/25 19:41:55 $
 *  $Revision: 1.13 $
 * \author Alex Tumanov 
 */

#include <FWCore/Framework/interface/EDProducer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include "CondFormats/CSCObjects/interface/CSCReadoutMappingFromFile.h"
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

  bool debug, PrintEventNumber, goodEvent, useExaminer, unpackStatusDigis, unpackMTCCData; 
  int numOfEvents;
  unsigned int errorMask, examinerMask;
  CSCReadoutMappingFromFile theMapping;
  bool instatiateDQM;
  CSCMonitorInterface * monitor;
  edm::InputTag inputObjectsTag; // input tag labelling raw data for input



};

#endif
