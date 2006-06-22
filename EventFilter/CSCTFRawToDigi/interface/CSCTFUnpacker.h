#ifndef CSCTFUnpacker_h
#define CSCTFUnpacker_h

/** \class CSCTFDCCUnpacker
 *
 *
 *  $Date: 2006/06/20 17:02:26 $
 *  $Revision: 1.2 $
 * \author Lindsey Gray
 */

#include <FWCore/Framework/interface/EDProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include "L1Trigger/CSCTrackFinder/interface/CSCTFPtLUT.h"

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

  CSCTFPtLUT ptlut;

  int numOfEvents;
  int TBFEDid,TBendcap,TBsector;

  bool instantiateDQM;
  bool testBeam;
  bool debug;
  CSCTriggerMappingFromFile* TFmapping;
  CSCTFMonitorInterface * monitor;

};

#endif
