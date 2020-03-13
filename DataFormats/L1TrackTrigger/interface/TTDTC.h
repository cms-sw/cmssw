#ifndef __L1TrackTrigger_TTDTC__
#define __L1TrackTrigger_TTDTC__

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
  // outer tracker DTC types
  typedef std::bitset<TTBV::S> BV;         // bit accurate Stub
  typedef std::pair<TTStubRef, BV> Frame;  // TTStub with bit accurate Stub
  typedef std::vector<Frame> Stream;       // stub collection transported over an optical link between DTC and TFP
  typedef std::vector<Stream> Streams;     // collection of optical links

public:
  TTDTC() {}

  TTDTC(const int& numRegions, const int& numOverlappingRegions, const int& numDTCsPerRegion);

  ~TTDTC() {}

  // Access to product configurations
  int numRegions() const;
  int numDTCBoards() const;
  int numDTCChannel() const;
  int numTFPChannel() const;

  // write one specific stream of TTStubRefs using DTC identifier (region[0-8], board[0-23], channel[0-1])
  // dtcRegions aka detector regions are defined by tk layout
  void setStream(const int& dtcRegion, const int& dtcBoard, const int& dtcChannel, const Stream& stream);

  // all TFP identifier (region[0-8], channel[0-47])
  std::vector<int> tfpRegions() const;
  std::vector<int> tfpChannels() const;

  // read one specific stream of TTStubRefs using TFP identifier (region[0-8], channel[0-47])
  // tfpRegions aka processing regions are rotated by -0.5 region width w.r.t detector regions
  const Stream& getStream(const int& tfpRegion, const int& tfpChannel) const;

  // converts dtc id into tk layout scheme
  int tkLayoutId(const int& dtcId) const;

  // converts tk layout id into dtc id
  int dtcId(const int& tkLayoutId) const;

private:
  // converts DTC identifier (region[0-8], board[0-23], channel[0-1]) into allStreams_ index [0-431]
  int index(const int& dtcRegion, const int& dtcBoard, const int& dtcChannel) const;

  // converts TFP identifier (region[0-8], channel[0-47]) into allStreams_ index [0-431]
  int index(const int& tfpRegion, const int& tfpChannel) const;

private:
  static constexpr int numSlots_ = 12;  // number of ATCA slots per shelf

  int numRegions_;             // number of phi slices the outer tracker readout is organized in [default 9]
  int numOverlappingRegions_;  // number of regions a reconstructable particle may cross [default 2]
  int numDTCsPerRegion_;       // number of DTC boards used to readout a detector region [default 24]
  int numDTCsPerTFP_;          // number of DTC boards connected to one TFP [default 48]

  Streams streams_;  // collection of all optical links between DTC and TFP [default 432 links]
};

#endif