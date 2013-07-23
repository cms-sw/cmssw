//-------------------------------------------------
//
/**  \class DTTrackFinder
 *
 *   L1 DT Track Finder EDProducer
 *
 *
 *   $Date: 2007/03/30 09:09:21 $
 *   $Revision: 1.3 $
 *
 *   J. Troconiz              UAM Madrid
 */
//
//--------------------------------------------------
#ifndef DTTrackFinder_h
#define DTTrackFinder_h

#include <FWCore/Framework/interface/EDProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <string>

class L1MuDTTFSetup;


class DTTrackFinder: public edm::EDProducer {
 public:
  /// Constructor
  DTTrackFinder(const edm::ParameterSet & pset);
  
  /// Destructor
  virtual ~DTTrackFinder();
  
  /// Produce digis out of raw data
  void produce(edm::Event & e, const edm::EventSetup& c);
  
 private:

  L1MuDTTFSetup* setup1;

};

#endif
