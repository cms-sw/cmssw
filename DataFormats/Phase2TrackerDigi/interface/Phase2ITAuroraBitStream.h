#ifndef DataFormats_Phase2TrackerDigi_Phase2ITAuroraBitStream_H
#define DataFormats_Phase2TrackerDigi_Phase2ITAuroraBitStream_H
#include <vector>

class Phase2ITAuroraBitStream {
  // Per elink Aurora bit streams, one stream per NE-event group
public:
  Phase2ITAuroraBitStream() : chipId_(-1), eventsPerStream_(0) {}
  Phase2ITAuroraBitStream(int chipId, int eventsPerStream) : chipId_(chipId), eventsPerStream_(eventsPerStream) {}

  void addAuroraStream(const std::vector<bool>& bits) { auroraStreams_.push_back(bits); }

  int get_chipId() const { return chipId_; }
  int get_eventsPerStream() const { return eventsPerStream_; }
  const std::vector<std::vector<bool>>& get_auroraStreams() const { return auroraStreams_; }

  const bool operator<(const Phase2ITAuroraBitStream& other) { return chipId_ < other.chipId_; }

private:
  int chipId_;                                    // Chip index within the module
  int eventsPerStream_;                           // NE: events per stream group
  std::vector<std::vector<bool>> auroraStreams_;  // full Aurora bits per stream group
};
#endif  // DataFormats_Phase2TrackerDigi_Phase2ITAuroraBitStream_H
