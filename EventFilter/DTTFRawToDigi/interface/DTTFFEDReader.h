//-------------------------------------------------
//
/**  \class DTTFFEDReader
 *
 *   L1 DT Track Finder Raw-to-Digi
 *
 *
 *   $Date: 2010/02/11 00:11:38 $
 *   $Revision: 1.8 $
 *
 *   J. Troconiz  UAM Madrid
 *   E. Delmeire  UAM Madrid
 */
//
//--------------------------------------------------
#ifndef DTTFRawToDigi_DTTFFEDReader_h
#define DTTFRawToDigi_DTTFFEDReader_h

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTTrackContainer.h"

#include <FWCore/Framework/interface/EDProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Utilities/interface/InputTag.h>

#include <string>

class DTTFFEDReader : public edm::EDProducer {

 public:

  /// Constructor
  DTTFFEDReader(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTTFFEDReader();

  /// Produce digis out of raw data
  void produce(edm::Event & e, const edm::EventSetup& c);

  /// Generate and fill FED raw data for a full event
  bool fillRawData(edm::Event& e,
		   L1MuDTChambPhContainer::Phi_Container& phi_data,
		   L1MuDTChambThContainer::The_Container& the_data,
		   L1MuDTTrackContainer::TrackContainer&  tra_data);

 private:
  
  edm::InputTag DTTFInputTag;

  bool verbose_;

  // Operations

  // access data
  const L1MuDTChambPhContainer::Phi_Container& p_data();

  const L1MuDTChambThContainer::The_Container& t_data();

  const L1MuDTTrackContainer::TrackContainer&  k_data();

  // Process one event
  void analyse(edm::Event& e);

  // clear data container
  void clear();

  // process data
  void process(edm::Event& e);

  // Match PHTF - ETTF tracks
  void match();

  // data containers
  L1MuDTChambPhContainer::Phi_Container phiSegments;

  L1MuDTChambThContainer::The_Container theSegments;

  L1MuDTTrackContainer::TrackContainer  dtTracks;

  unsigned int etTrack[3][12][6][2];

  unsigned int efTrack[3][12][6][2];

  // utilities
  int channel(int wheel, int sector, int bx);

  int bxNr(int channel);

  int sector(int channel);

  int wheel(int channel);

  void calcCRC(int myD1, int myD2, int &myC);

  edm::InputTag getDTTFInputTag() { return DTTFInputTag; }

};
#endif
