#ifndef CSCTrackFinder_CSCTFTrackBuilder_h
#define CSCTrackFinder_CSCTFTrackBuilder_h

#include <vector>
#include <DataFormats/L1CSCTrackFinder/interface/L1Track.h>
#include <DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>

#include <L1Trigger/CSCTrackFinder/interface/CSCTFSectorProcessor.h>

#include <FWCore/ParameterSet/interface/ParameterSet.h>

class CSCTFTrackBuilder
{
 public:

  enum { nEndcaps = 2, nSectors = 6};

  CSCTFTrackBuilder(const edm::ParameterSet& pset);

  ~CSCTFTrackBuilder();

  void buildTracks(const CSCCorrelatedLCTDigiCollection*, std::vector<csc::L1Track>*);
  
 private:


  CSCTFSectorProcessor* my_SPs[nEndcaps][nSectors];
};

#endif
