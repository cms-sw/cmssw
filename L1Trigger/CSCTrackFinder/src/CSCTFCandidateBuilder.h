/**
 * \class CSCTFCandidateBuilder
 * \author L. Gray (UF) 
 *
 * Takes sorts csc::L1Tracks and turns them into L1MuRegionalCands
 */

#ifndef CSCTrackFinder_CSCTFCandidateBuilder_h
#define CSCTrackFinder_CSCTFCandidateBuilder_h

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/CSCTrackFinder/interface/CSCTFMuonSorter.h"

class CSCTFCandidateBuilder
{
 public:
  explicit CSCTFCandidateBuilder(const edm::ParameterSet&);

  void buildCandidates(const L1CSCTrackCollection*, std::vector<L1MuRegionalCand>*) const;

 private:
  
  CSCTFMuonSorter m_muonsorter;
};

#endif
