#ifndef DataFormats_Phase2TrackerDigi_Phase2ITAuroraBitStream_H
#define DataFormats_Phase2TrackerDigi_Phase2ITAuroraBitStream_H
#include <vector>

class Phase2ITAuroraBitStream {
  // Per elink Aurora bit streams, one stream per NE-event group
public:
  Phase2ITAuroraBitStream() : elinkId_(-1), eventsPerStream_(0) {}
  Phase2ITAuroraBitStream(int elinkId, int eventsPerStream) : elinkId_(elinkId), eventsPerStream_(eventsPerStream) {}

  void addAuroraStream(const std::vector<bool>& bits) { auroraStreams_.push_back(bits); }

  int get_elinkId() const { return elinkId_; }
  int get_eventsPerStream() const { return eventsPerStream_; }
  const std::vector<std::vector<bool>>& get_auroraStreams() const { return auroraStreams_; }

  bool operator<(const Phase2ITAuroraBitStream& other) const { return elinkId_ < other.elinkId_; }

private:
  int elinkId_;                                   // ELink index within the module
  int eventsPerStream_;                           // NE: events per stream group
  std::vector<std::vector<bool>> auroraStreams_;  // full Aurora bits per stream group
};
#endif  // DataFormats_Phase2TrackerDigi_Phase2ITAuroraBitStream_H
