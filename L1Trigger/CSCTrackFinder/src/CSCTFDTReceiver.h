//
// Class: CSCTFDTReceiver
// Use: Takes a set of DT track stubs and converts them into
//      CSC Track Stubs. Emulates all of the DT receiver cards in a TF crate.
// Author: Lindsey Gray (partial port from ORCA)
//

#include <DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h>
#include <DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h>
#include <DataFormats/L1CSCTrackFinder/interface/TrackStub.h>

class CSCTFDTReceiver
{
 public:

  CSCTFDTReceiver() { dtstubs.clear(); }
  ~CSCTFDTReceiver() {}

  // Takes input DT Sector Collector stubs and translates them into CSC coordinates.
  CSCTriggerContainer<csctf::TrackStub> process(const L1MuDTChambPhContainer*);
  

 private:
  CSCTriggerContainer<csctf::TrackStub> dtstubs;

};
