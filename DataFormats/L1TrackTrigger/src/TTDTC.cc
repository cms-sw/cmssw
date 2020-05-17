#include "DataFormats/L1TrackTrigger/interface/TTDTC.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <numeric>

using namespace std;
using namespace edm;

TTDTC::TTDTC(int numRegions, int numOverlappingRegions, int numDTCsPerRegion)
    : numRegions_(numRegions),
      numOverlappingRegions_(numOverlappingRegions),
      numDTCsPerRegion_(numDTCsPerRegion),
      numDTCsPerTFP_(numOverlappingRegions * numDTCsPerRegion),
      regions_(numRegions_),
      channels_(numDTCsPerRegion_ * numOverlappingRegions_),
      streams_(numRegions_ * channels_.size()) {
  iota(regions_.begin(), regions_.end(), 0);
  iota(channels_.begin(), channels_.end(), 0);
}

// write one specific stream of TTStubRefs using DTC identifier (region[0-8], board[0-23], channel[0-1])
// dtcRegions aka detector regions are defined by tk layout
void TTDTC::setStream(int dtcRegion, int dtcBoard, int dtcChannel, const Stream& stream) {
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
const TTDTC::Stream& TTDTC::stream(int tfpRegion, int tfpChannel) const {
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
  auto all = [](int& sum, const Stream& stream) { return sum += stream.size(); };
  return accumulate(streams_.begin(), streams_.end(), 0, all);
}

// total number of stubs
int TTDTC::nStubs() const {
  auto stubs = [](int& sum, const Frame& frame) { return sum += frame.first.isNonnull(); };
  int n(0);
  for (const Stream& stream : streams_)
    n += accumulate(stream.begin(), stream.end(), 0, stubs);
  return n;
}

// total number of gaps
int TTDTC::nGaps() const {
  auto gaps = [](int& sum, const Frame& frame) { return sum += frame.first.isNull(); };
  int n(0);
  for (const Stream& stream : streams_)
    n += accumulate(stream.begin(), stream.end(), 0, gaps);
  return n;
}

// converts dtc id into tk layout scheme
int TTDTC::tkLayoutId(int dtcId) const {
  // check argument
  if (dtcId < 0 || dtcId >= numDTCsPerRegion_ * numRegions_) {
    cms::Exception exception("out_of_range");
    exception.addContext("TTDTC::tkLayoutId");
    exception << "Used DTC Id (" << dtcId << ") is out of range 0 to " << numDTCsPerRegion_ * numRegions_ - 1 << ".";
    throw exception;
  }
  const int slot = dtcId % numSlots_;
  const int region = dtcId / numDTCsPerRegion_;
  const int side = (dtcId % numDTCsPerRegion_) / numSlots_;
  return (side * numRegions_ + region) * numSlots_ + slot + 1;
}

// converts tk layout id into dtc id
int TTDTC::dtcId(int tkLayoutId) const {
  // check argument
  if (tkLayoutId <= 0 || tkLayoutId > numDTCsPerRegion_ * numRegions_) {
    cms::Exception exception("out_of_range");
    exception.addContext("TTDTC::dtcId");
    exception << "Used TKLayout Id (" << tkLayoutId << ") is out of range 1 to " << numDTCsPerRegion_ * numRegions_
              << ".";
    throw exception;
  }
  const int tkId = tkLayoutId - 1;
  const int side = tkId / (numRegions_ * numSlots_);
  const int region = (tkId % (numRegions_ * numSlots_)) / numSlots_;
  const int slot = tkId % numSlots_;
  return region * numDTCsPerRegion_ + side * numSlots_ + slot;
}

// converts TFP identifier (region[0-8], channel[0-47]) into dtc id
int TTDTC::dtcId(int tfpRegion, int tfpChannel) const { return index(tfpRegion, tfpChannel) / numOverlappingRegions_; }

// checks if given DTC id is connected to PS or 2S sensormodules
bool TTDTC::psModlue(int dtcId) const {
  // from tklayout: first 3 are 10 gbps PS, next 3 are 5 gbps PS and residual 6 are 5 gbps 2S modules
  return slot(dtcId) < numSlots_ / 2;
}

// checks if given dtcId is connected to -z (false) or +z (true)
bool TTDTC::side(int dtcId) const {
  const int side = (dtcId % numDTCsPerRegion_) / numSlots_;
  // from tkLayout: first 12 +z, next 12 -z
  return side == 0;
}
// ATCA slot number [0-11] of given dtcId
int TTDTC::slot(int dtcId) const { return dtcId % numSlots_; }

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