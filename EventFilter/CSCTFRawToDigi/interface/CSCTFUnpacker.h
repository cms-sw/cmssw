#ifndef CSCTFUnpacker_h
#define CSCTFUnpacker_h

/** \class CSCTFDCCUnpacker
 * 
 *
 *  $Date: 2005/12/07 15:01:53 $
 *  $Revision: 1.1 $
 * \author Lindsey Gray 
 */

#include <FWCore/Framework/interface/EDProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <string>

class CSCTFMonitorInterface;
class CSCTriggerMappingFromFile;

class CSCTFUnpacker: public edm::EDProducer {
 public:
  /// Constructor
  CSCTFUnpacker(const edm::ParameterSet & pset);
  
  /// Destructor
  virtual ~CSCTFUnpacker();
  
  /// Produce digis out of raw data
  void produce(edm::Event & e, const edm::EventSetup& c);



  
 private:
 
  int numOfEvents;
  int TBFEDid,TBendcap,TBsector;
  
  bool instantiateDQM;
  bool testBeam;
  bool debug;
  CSCTriggerMappingFromFile* TFmapping;
  CSCTFMonitorInterface * monitor;


};

#endif
