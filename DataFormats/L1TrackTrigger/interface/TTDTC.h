#ifndef DataFormats_L1TrackTrigger_TTDTC_h
#define DataFormats_L1TrackTrigger_TTDTC_h

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTBV.h"

#include <bitset>
#include <utility>
#include <vector>

/*! 
 * \class  TTDTC
 * \brief  Class to store hardware like structured TTStub Collection used by Track Trigger emulators
 * \author Thomas Schuh
 * \date   2020, Jan
 */
class TTDTC {
public:
  // bit accurate Stub
  typedef std::bitset<TTBV::S> BV;
  // TTStub with bit accurate Stub
  typedef std::pair<TTStubRef, BV> Frame;
  // stub collection transported over an optical link between DTC and TFP
  typedef std::vector<Frame> Stream;
  // collection of optical links
  typedef std::vector<Stream> Streams;

private:
  // number of ATCA slots per shelf
  static constexpr int numSlots_ = 12;

public:
  TTDTC() {}
  TTDTC(int numRegions, int numOverlappingRegions, int numDTCsPerRegion);
  ~TTDTC() {}

  // number of phi slices the outer tracker readout is organized in [default 9]
  int numRegions() const { return numRegions_; }
  // number of DTC boards used to readout a detector region [default 24]
  int numDTCBoards() const { return numDTCsPerRegion_; }
  // number of regions a reconstructable particle may cross [default 2]
  int numDTCChannel() const { return numOverlappingRegions_; }
  // number of DTC boards connected to one TFP [default 48]
  int numTFPChannel() const { return numDTCsPerTFP_; }
  // all regions [default 0..8]
  const std::vector<int>& tfpRegions() const { return regions_; }
  // all TFP channel [default 0..47]
  const std::vector<int>& tfpChannels() const { return channels_; }
  // write one specific stream of TTStubRefs using DTC identifier (region[0-8], board[0-23], channel[0-1])
  // dtcRegions aka detector regions are defined by tk layout
  void setStream(int dtcRegion, int dtcBoard, int dtcChannel, const Stream& stream);
  // read one specific stream of TTStubRefs using TFP identifier (region[0-8], channel[0-47])
  // tfpRegions aka processing regions are rotated by -0.5 region width w.r.t detector regions
  const Stream& stream(int tfpRegion, int tfpChannel) const;
  // total number of frames
  int size() const;
  // total number of stubs
  int nStubs() const;
  // total number of gaps
  int nGaps() const;
  // converts dtc id into tk layout scheme
  int tkLayoutId(int dtcId) const;
  // converts tk layout id into dtc id
  int dtcId(int tkLayoutId) const;
  // converts TFP identifier (region[0-8], channel[0-47]) into dtcId [0-215]
  int dtcId(int tfpRegion, int tfpChannel) const;
  // checks if given dtcId is connected to PS or 2S sensormodules
  bool psModlue(int dtcId) const;
  // checks if given dtcId is connected to -z (false) or +z (true)
  bool side(int dtcId) const;
  // ATCA slot number [0-11] of given dtcId
  int slot(int dtcId) const;

private:
  // converts DTC identifier (region[0-8], board[0-23], channel[0-1]) into allStreams_ index [0-431]
  int index(int dtcRegion, int dtcBoard, int dtcChannel) const;
  // converts TFP identifier (region[0-8], channel[0-47]) into allStreams_ index [0-431]
  int index(int tfpRegion, int tfpChannel) const;
  // number of phi slices the outer tracker readout is organized in [default 9]
  int numRegions_;
  // number of regions a reconstructable particle may cross [default 2]
  int numOverlappingRegions_;
  // number of DTC boards used to readout a detector region [default 24]
  int numDTCsPerRegion_;
  // number of DTC boards connected to one TFP [default 48]
  int numDTCsPerTFP_;
  // all regions [default 0..8]
  std::vector<int> regions_;
  // all TFP channel [default 0..47]
  std::vector<int> channels_;
  // collection of all optical links between DTC and TFP [default 432 links]
  Streams streams_;
};

#endif