#ifndef CSCTrackFinder_CSCTFTrackBuilder_h
#define CSCTrackFinder_CSCTFTrackBuilder_h

#include <vector>
#include <DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h>
#include <DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h>
#include <DataFormats/L1CSCTrackFinder/interface/CSCTrackStub.h>
#include <DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h>

#include <FWCore/ParameterSet/interface/ParameterSet.h>

class CSCMuonPortCard;
class CSCTFSectorProcessor;

class CSCTFTrackBuilder
{
 public:

  enum { nEndcaps = 2, nSectors = 6};

  CSCTFTrackBuilder(const edm::ParameterSet& pset);

  ~CSCTFTrackBuilder();

  void buildTracks(const CSCCorrelatedLCTDigiCollection*, L1CSCTrackCollection*, 
		   CSCTriggerContainer<CSCTrackStub>*);
  
 private:

  CSCTFSectorProcessor* my_SPs[nEndcaps][nSectors];
};

#endif
