#ifndef CSCDCCUnpacker_h
#define CSCDCCUnpacker_h

/** \class CSCDCCUnpacker
 * 
 *
 *  $Date: 2010/06/11 15:50:27 $
 *  $Revision: 1.25 $
 * \author Alex Tumanov 
 */

#include <FWCore/Framework/interface/EDProducer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include "FWCore/Utilities/interface/InputTag.h"

class CSCMonitorInterface;

class CSCDCCUnpacker: public edm::EDProducer {
 public:
  /// Constructor
  CSCDCCUnpacker(const edm::ParameterSet & pset);
  
  /// Destructor
  virtual ~CSCDCCUnpacker();
  
  /// Produce digis out of raw data
  void produce(edm::Event & e, const edm::EventSetup& c);
  
  /// Visualization of raw data in FED-less events (Robert Harr and Alexander Sakharov)
  void visual_raw(int hl,int id, int run, int event, bool fedshort, bool fDump, short unsigned int* buf) const; 
  
 private:

  bool debug, printEventNumber, goodEvent, useExaminer, unpackStatusDigis;
  bool useSelectiveUnpacking, useFormatStatus;
  
  /// Visualization of raw data
  bool  visualFEDInspect, visualFEDShort, formatedEventDump;
  /// Suppress zeros LCTs
  bool SuppressZeroLCT;
  
  int numOfEvents;
  unsigned int errorMask, examinerMask;
  bool instantiateDQM;
  CSCMonitorInterface * monitor;
  edm::InputTag inputObjectsTag; // input tag labelling raw data for input



};

#endif
