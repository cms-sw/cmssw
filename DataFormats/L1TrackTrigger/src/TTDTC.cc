#include "DataFormats/L1TrackTrigger/interface/TTDTC.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <numeric>

using namespace std;
using namespace edm;
using namespace tt;

TTDTC::TTDTC(int numRegions, int numOverlappingRegions, int numDTCsPerRegion)
    : numRegions_(numRegions),
      numOverlappingRegions_(numOverlappingRegions),
      numDTCsPerRegion_(numDTCsPerRegion),
      numDTCsPerTFP_(numOverlappingRegions * numDTCsPerRegion),
      regions_(numRegions_),
      channels_(numDTCsPerTFP_),
      streams_(numRegions_ * numDTCsPerTFP_) {
  iota(regions_.begin(), regions_.end(), 0);
  iota(channels_.begin(), channels_.end(), 0);
}

// write one specific stream of TTStubRefs using DTC identifier (region[0-8], board[0-23], channel[0-1])
// dtcRegions aka detector regions are defined by tk layout
void TTDTC::setStream(int dtcRegion, int dtcBoard, int dtcChannel, const StreamStub& stream) {
  // check arguments
  const bool oorRegion = dtcRegion >= numRegions_ || dtcRegion < 0;
  const bool oorBoard = dtcBoard >= numDTCsPerRegion_ || dtcBoard < 0;
  const bool oorChannel = dtcChannel >= numOverlappingRegions_ || dtcChannel < 0;
  if (oorRegion || oorBoard || oorChannel) {
    cms::Exception exception("out_of_range");
    exception.addContext("TTDTC::setStream");
    if (oorRegion)
      exception << "Requested Detector Region "
                << "(" << dtcRegion << ") is out of range 0 to " << numRegions_ - 1 << ".";
    if (oorBoard)
      exception << "Requested DTC Board "
                << "(" << dtcBoard << ") is out of range 0 to " << numDTCsPerRegion_ - 1 << ".";
    if (oorChannel)
      exception << "Requested DTC Channel "
                << "(" << dtcChannel << ") is out of range 0 to " << numOverlappingRegions_ - 1 << ".";
    throw exception;
  }
  streams_[index(dtcRegion, dtcBoard, dtcChannel)] = stream;
}

// read one specific stream of TTStubRefs using TFP identifier (region[0-8], channel[0-47])
// tfpRegions aka processing regions are rotated by -0.5 region width w.r.t detector regions
const StreamStub& TTDTC::stream(int tfpRegion, int tfpChannel) const {
  // check arguments
  const bool oorRegion = tfpRegion >= numRegions_ || tfpRegion < 0;
  const bool oorChannel = tfpChannel >= numDTCsPerTFP_ || tfpChannel < 0;
  if (oorRegion || oorChannel) {
    cms::Exception exception("out_of_range");
    exception.addContext("TTDTC::stream");
    if (oorRegion)
      exception << "Requested Processing Region "
                << "(" << tfpRegion << ") is out of range 0 to " << numRegions_ - 1 << ".";
    if (oorChannel)
      exception << "Requested TFP Channel "
                << "(" << tfpChannel << ") is out of range 0 to " << numDTCsPerTFP_ - 1 << ".";
    throw exception;
  }
  return streams_.at(index(tfpRegion, tfpChannel));
}

// total number of frames
int TTDTC::size() const {
  auto all = [](int sum, const StreamStub& stream) { return sum + stream.size(); };
  return accumulate(streams_.begin(), streams_.end(), 0, all);
}

// total number of stubs
int TTDTC::nStubs() const {
  auto stubs = [](int sum, const FrameStub& frame) { return sum + frame.first.isNonnull(); };
  int n(0);
  for (const StreamStub& stream : streams_)
    n += accumulate(stream.begin(), stream.end(), 0, stubs);
  return n;
}

// total number of gaps
int TTDTC::nGaps() const {
  auto gaps = [](int sum, const FrameStub& frame) { return sum + frame.first.isNull(); };
  int n(0);
  for (const StreamStub& stream : streams_)
    n += accumulate(stream.begin(), stream.end(), 0, gaps);
  return n;
}

// converts DTC identifier (region[0-8], board[0-23], channel[0-1]) into streams_ index [0-431]
int TTDTC::index(int dtcRegion, int dtcBoard, int dtcChannel) const {
  return (dtcRegion * numDTCsPerRegion_ + dtcBoard) * numOverlappingRegions_ + dtcChannel;
}

// converts TFP identifier (region[0-8], channel[0-47]) into streams_ index [0-431]
int TTDTC::index(int tfpRegion, int tfpChannel) const {
  const int dtcChannel = numOverlappingRegions_ - (tfpChannel / numDTCsPerRegion_) - 1;
  const int dtcBoard = tfpChannel % numDTCsPerRegion_;
  const int dtcRegion = tfpRegion - dtcChannel >= 0 ? tfpRegion - dtcChannel : tfpRegion - dtcChannel + numRegions_;
  return index(dtcRegion, dtcBoard, dtcChannel);
}
