#ifndef L1Trigger_TrackFindingTracklet_interface_PurgeDuplicate_h
#define L1Trigger_TrackFindingTracklet_interface_PurgeDuplicate_h

#include "L1Trigger/TrackFindingTracklet/interface/TrackFitMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/CleanTrackMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/ProcessBase.h"

#include <vector>

// Run algorithm to remove duplicate tracks.
// Returns Track object that represent output at end of L1 track chain,
// (and are later converted to TTTrack). Duplicate Track objects are flagged,
// but only deleted if using the Hybrid algo.
// Also writes to memory the same tracks in more detailed Tracklet format,
// where duplicates are all deleted.

namespace trklet {

  class Settings;
  class Globals;
  class MemoryBase;
  class Stub;
  class L1TStub;
  class Track;
  class Tracklet;

  class PurgeDuplicate : public ProcessBase {
  public:
    PurgeDuplicate(std::string name, Settings const& settings, Globals* global);

    ~PurgeDuplicate() override = default;

    void addOutput(MemoryBase* memory, std::string output) override;
    void addInput(MemoryBase* memory, std::string input) override;

    void execute(std::vector<Track>& outputtracks_, unsigned int iSector);

  private:
    double getPhiRes(Tracklet* curTracklet, const Stub* curStub) const;
    bool isSeedingStub(int, int, int) const;
    std::string l1tinfo(const L1TStub*, std::string) const;
    std::pair<int, int> findLayerDisk(const Stub*) const;
    // calculate stub coordinates based on tracklet trajectory for prompt tracking
    std::vector<double> getInventedCoords(unsigned int, const Stub*, const Tracklet*) const;
    // calculate stub coordinates based on tracklet trajectory for extended tracking
    std::vector<double> getInventedCoordsExtended(unsigned int, const Stub*, const Tracklet*) const;
    // return stub with invented x,y,z coords, if it's a seeding one for this tracklet
    std::vector<const Stub*> getInventedSeedingStub(unsigned int,
                                                    const Tracklet*,
                                                    const std::vector<const Stub*>&) const;
    // return the regular rinvbins which contain the input tracklet
    unsigned int findVarRInvBin(const Tracklet* trk) const;
    // return the overlap rinvbins which contain the input tracklet
    std::vector<unsigned int> findOverlapRInvBins(const Tracklet* trk) const;
    // sort the tracklets into the correct bin by comparing the overlap rinv bin(s) the tracklets are in to the current bin
    bool isTrackInBin(const std::vector<unsigned int>& vec, unsigned int num) const;

    std::vector<Track*> inputtracks_;
    std::vector<std::vector<const Stub*>> inputstublists_;
    std::vector<std::vector<std::pair<int, int>>> inputstubidslists_;
    std::vector<std::vector<std::pair<int, int>>> mergedstubidslists_;
    std::vector<TrackFitMemory*> inputtrackfits_;
    std::vector<Tracklet*> inputtracklets_;
    std::vector<CleanTrackMemory*> outputtracklets_;
  };

};  // namespace trklet
#endif
