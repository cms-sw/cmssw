#ifndef CSCDCCUnpacker_h
#define CSCDCCUnpacker_h

/** \class CSCDCCUnpacker
 * 
 *
 *  $Date: 2009/03/27 10:53:13 $
 *  $Revision: 1.19 $
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
  
  /// Visualization of raw data in FED-less events
  void visual_raw(int hl,int id, int run, int event, bool fedshort, short unsigned int* buf) const; 
  // Visualization of raw data in FED-less events

  
 private:

  bool debug, printEventNumber, goodEvent, useExaminer, unpackStatusDigis;
  bool useSelectiveUnpacking, useFormatStatus;
  // Visualization of raw data in FED-less events
  bool  visualFEDInspect, visualFEDShort;
  // Visualization of raw data in FED-less events
  int numOfEvents;
  unsigned int errorMask, examinerMask;
  bool instatiateDQM;
  CSCMonitorInterface * monitor;
  edm::InputTag inputObjectsTag; // input tag labelling raw data for input



};

#endif
