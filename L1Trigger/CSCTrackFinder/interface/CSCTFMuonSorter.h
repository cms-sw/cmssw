#ifndef CSCTrackFinder_CSCTFMuonSorter_h
#define CSCTrackFinder_CSCTFMuonSorter_h

#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h>
#include <DataFormats/L1CSCTrackFinder/interface/L1Track.h>
#include <DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h>

class CSCTFMuonSorter
{
 public:
  CSCTFMuonSorter(const edm::ParameterSet&);

  std::vector<L1MuRegionalCand> run(const CSCTriggerContainer<csc::L1Track>&) const;

 private:
  void decodeRank(const unsigned& rank, unsigned& quality, unsigned& pt) const;

  int m_minBX, m_maxBX;
};

#endif
