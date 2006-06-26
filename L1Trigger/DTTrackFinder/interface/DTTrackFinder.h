//-------------------------------------------------
//
/**  \class DTTrackFinder
 *
 *   L1 DT Track Finder EDProducer
 *
 *
 *   $Date: 2006/06/01 00:00:00 $
 *   $Revision: 1.1 $
 *
 *   J. Troconiz              UAM Madrid
 */
//
//--------------------------------------------------
#ifndef DTTrackFinder_h
#define DTTrackFinder_h

/** \class DTTrackFinder
 * 
 *
 *  $Date: 2006/06/01 00:00:00 $
 *  $Revision: 1.1 $
 *
 *          Jorge Troconiz  UAM Madrid
 */

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
