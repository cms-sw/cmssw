#ifndef CSCDCCUnpacker_h
#define CSCDCCUnpacker_h

/** \class CSCDCCUnpacker
 * 
 *
 *  $Date: 2006/02/22 09:39:30 $
 *  $Revision: 1.8 $
 * \author Alex Tumanov 
 */

#include <FWCore/Framework/interface/EDProducer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include "CondFormats/CSCObjects/interface/CSCReadoutMappingFromFile.h"


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

  bool debug, PrintEventNumber, goodEvent, useExaminer; 
  int numOfEvents;
  CSCReadoutMappingFromFile theMapping;
  bool instatiateDQM;
  CSCMonitorInterface * monitor;


};

#endif
