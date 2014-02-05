//-------------------------------------------------
//
/**  \class DTTrackFinder
 *
 *   L1 DT Track Finder EDProducer
 *
 *
 *
 *   J. Troconiz              UAM Madrid
 */
//
//--------------------------------------------------
#ifndef DTTrackFinder_h
#define DTTrackFinder_h

#include <FWCore/Framework/interface/one/EDProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <string>

class L1MuDTTFSetup;


class DTTrackFinder: public edm::one::EDProducer<edm::one::SharedResources> {
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
