/**
 * \author L. Gray
 * \class CSCTFSectorProcessor.h
 * 
 * A class that represents a sector processor board.
 */

#ifndef CSCTrackFinder_CSCTFSectorProcessor_h
#define CSCTrackFinder_CSCTFSectorProcessor_h

#include <vector>
#include <map>
#include <string>
#include <DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h>
#include <DataFormats/L1CSCTrackFinder/interface/TrackStub.h>
#include <DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h>
#include <L1Trigger/CSCTrackFinder/interface/CSCTFSPCoreLogic.h>
#include <L1Trigger/CSCTrackFinder/interface/CSCTFPtLUT.h>

class CSCTFSectorProcessor
{
 public:
  CSCTFSectorProcessor(const unsigned& endcap, const unsigned& sector, const edm::ParameterSet&);
  
  ~CSCTFSectorProcessor();

  bool run(const CSCTriggerContainer<csctf::TrackStub>&);

  CSCTriggerContainer<csc::L1Track> tracks() const { return l1_tracks; }

  CSCTriggerContainer<csctf::TrackStub> dtStubs() const { return dt_stubs; }

  int minBX() const { return m_minBX; }
  int maxBX() const { return m_maxBX; }

 private:
  // disallow copy and assignment
  CSCTFSectorProcessor& operator=(const CSCTFSectorProcessor& rhs) { return *this; };
  CSCTFSectorProcessor(const CSCTFSectorProcessor& par) {}

  unsigned m_endcap, m_sector;
  unsigned m_bxa_on, m_extend_length;
  unsigned m_latency;

  int m_minBX, m_maxBX;
  unsigned m_etawin[6];
  
  CSCTriggerContainer<csc::L1Track> l1_tracks; // fully defined L1Tracks
  CSCTriggerContainer<csctf::TrackStub> dt_stubs; // Track Stubs to be sent to the DTTF

  static const std::string FPGAs[5];

  std::map<std::string, CSCSectorReceiverLUT*> srLUTs_; // indexed by FPGA
  CSCTFSPCoreLogic* core_;
  CSCTFPtLUT* ptLUT_;
};

#endif
