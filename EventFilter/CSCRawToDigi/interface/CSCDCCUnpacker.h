#ifndef CSCDCCUnpacker_h
#define CSCDCCUnpacker_h

/** \class CSCDCCUnpacker
 * 
 *
 *  $Date: 2005/11/29 18:06:48 $
 *  $Revision: 1.3 $
 * \author Alex Tumanov 
 */

#include <FWCore/Framework/interface/EDProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <EventFilter/CSCRawToDigi/interface/MonitorInterface.h>


class CSCDCCUnpacker: public edm::EDProducer {
 public:
  /// Constructor
  CSCDCCUnpacker(const edm::ParameterSet & pset);
  
  /// Destructor
  virtual ~CSCDCCUnpacker();
  
  /// Produce digis out of raw data
  void produce(edm::Event & e, const edm::EventSetup& c);



  
 private:

  int numOfEvents;
  
  bool instatiateDQM;
  MonitorInterface * monitor;

  
};

#endif
