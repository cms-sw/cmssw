#ifndef CSCTrackFinder_CSCTFTrackBuilder_h
#define CSCTrackFinder_CSCTFTrackBuilder_h

#include <vector>
#include <DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h>
#include <DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h>
#include <DataFormats/L1CSCTrackFinder/interface/TrackStub.h>
#include <DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h>
#include <DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h>
#include <cstring>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include "L1Trigger/CSCTrackFinder/interface/CSCTFSectorProcessor.h"

class CSCMuonPortCard;
class L1MuTriggerScales;
class L1MuTriggerPtScale;

class CSCTFTrackBuilder {
public:
  using Tokens = CSCTFSectorProcessor::Tokens;
  static Tokens consumes(const edm::ParameterSet& pset, edm::ConsumesCollector iC) {
    return CSCTFSectorProcessor::consumes(pset, iC);
  }

  void initialize(const edm::EventSetup& c, const Tokens& tokens);

  enum { nEndcaps = 2, nSectors = 6 };

  CSCTFTrackBuilder(const edm::ParameterSet& pset,
                    bool TMB07,
                    const L1MuTriggerScales* scales,
                    const L1MuTriggerPtScale* ptScale);

  ~CSCTFTrackBuilder();

  void buildTracks(const CSCCorrelatedLCTDigiCollection*,
                   const CSCTriggerContainer<csctf::TrackStub>*,  //const L1MuDTChambPhContainer*,
                   L1CSCTrackCollection*,
                   CSCTriggerContainer<csctf::TrackStub>*);

private:
  CSCTFSectorProcessor* my_SPs[nEndcaps][nSectors];
  int m_minBX, m_maxBX;
};

#endif
