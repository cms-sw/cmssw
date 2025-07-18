#ifndef L1Trigger_Phase2L1GMT_SAMuonCleaner_h
#define L1Trigger_Phase2L1GMT_SAMuonCleaner_h

#include "DataFormats/L1TMuonPhase2/interface/SAMuon.h"

class SAMuonCleaner {
public:
  SAMuonCleaner() = default;
  ~SAMuonCleaner() = default;

  std::vector<l1t::SAMuon> cleanTFMuons(const std::vector<l1t::SAMuon>& muons);

private:
  std::vector<l1t::SAMuon> cleanTF(const std::vector<l1t::SAMuon>& tfMuons);
  void overlapCleanTrack(l1t::SAMuon& source, const l1t::SAMuon& other, bool eq);
  void overlapCleanTrackInter(l1t::SAMuon& source, const l1t::SAMuon& other);
  std::vector<l1t::SAMuon> interTFClean(const std::vector<l1t::SAMuon>& bmtf,
                                        const std::vector<l1t::SAMuon>& omtf,
                                        const std::vector<l1t::SAMuon>& emtf);
  void swap(std::vector<l1t::SAMuon>&, int i, int j);
  void sort(std::vector<l1t::SAMuon>& in);
};

#endif
