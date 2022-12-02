#ifndef RecoTracker_MkFitCore_standalone_Event_h
#define RecoTracker_MkFitCore_standalone_Event_h

#include "RecoTracker/MkFitCore/interface/Track.h"
#include "Validation.h"
#include "RecoTracker/MkFitCore/interface/Config.h"

#include <mutex>

namespace mkfit {

  struct DataFile;

  class Event {
  public:
    explicit Event(int evtID, int nLayers);
    Event(Validation &v, int evtID, int nLayers);

    void reset(int evtID);
    void validate();
    void printStats(const TrackVec &, TrackExtraVec &);

    int evtID() const { return evtID_; }
    void resetLayerHitMap(bool resetSimHits);

    void write_out(DataFile &data_file);
    void read_in(DataFile &data_file, FILE *in_fp = 0);
    int write_tracks(FILE *fp, const TrackVec &tracks);
    int read_tracks(FILE *fp, TrackVec &tracks, bool skip_reading = false);

    void setInputFromCMSSW(std::vector<HitVec> hits, TrackVec seeds);

    void kludge_cms_hit_errors();

    int use_seeds_from_cmsswtracks();  //special mode --> use only seeds which generated cmssw reco track
    int clean_cms_simtracks();
    int clean_cms_seedtracks(
        TrackVec *seed_ptr = nullptr);    //operates on seedTracks_; returns the number of cleaned seeds
    int clean_cms_seedtracks_badlabel();  //operates on seedTracks_, removes those with label == -1;
    void relabel_bad_seedtracks();
    void relabel_cmsswtracks_from_seeds();

    int select_tracks_iter(unsigned int n = 0);  //for cmssw input

    void fill_hitmask_bool_vectors(int track_algo, std::vector<std::vector<bool>> &layer_masks);
    void fill_hitmask_bool_vectors(std::vector<int> &track_algo_vec, std::vector<std::vector<bool>> &layer_masks);

    void print_tracks(const TrackVec &tracks, bool print_hits) const;

    Validation &validation_;

  private:
    int evtID_;

  public:
    BeamSpot beamSpot_;  // XXXX Read/Write of BeamSpot + file-version bump or extra-section to be added.
    std::vector<HitVec> layerHits_;
    std::vector<std::vector<uint64_t>> layerHitMasks_;  //aligned with layerHits_
    MCHitInfoVec simHitsInfo_;

    TrackVec simTracks_, seedTracks_, candidateTracks_, fitTracks_;
    TrackVec cmsswTracks_;
    // validation sets these, so needs to be mutable
    mutable TrackExtraVec simTracksExtra_, seedTracksExtra_, candidateTracksExtra_, fitTracksExtra_;
    mutable TrackExtraVec cmsswTracksExtra_;

    TSVec simTrackStates_;
    static std::mutex printmutex;
  };

  typedef std::vector<Event> EventVec;

  struct DataFileHeader {
    int f_magic = 0xBEEF;
    int f_format_version = 7;  //last update with ph2 geom
    int f_sizeof_track = sizeof(Track);
    int f_sizeof_hit = sizeof(Hit);
    int f_sizeof_hot = sizeof(HitOnTrack);
    int f_n_layers = -1;
    int f_n_events = -1;

    int f_extra_sections = 0;

    DataFileHeader() = default;
  };

  struct DataFile {
    enum ExtraSection {
      ES_SimTrackStates = 0x1,
      ES_Seeds = 0x2,
      ES_CmsswTracks = 0x4,
      ES_HitIterMasks = 0x8,
      ES_BeamSpot = 0x10
    };

    FILE *f_fp = 0;
    long f_pos = sizeof(DataFileHeader);

    DataFileHeader f_header;

    std::mutex f_next_ev_mutex;

    // ----------------------------------------------------------------

    bool hasSimTrackStates() const { return f_header.f_extra_sections & ES_SimTrackStates; }
    bool hasSeeds() const { return f_header.f_extra_sections & ES_Seeds; }
    bool hasCmsswTracks() const { return f_header.f_extra_sections & ES_CmsswTracks; }
    bool hasHitIterMasks() const { return f_header.f_extra_sections & ES_HitIterMasks; }
    bool hasBeamSpot() const { return f_header.f_extra_sections & ES_BeamSpot; }

    int openRead(const std::string &fname, int expected_n_layers);
    void openWrite(const std::string &fname, int n_layers, int n_ev, int extra_sections = 0);

    void rewind();

    int advancePosToNextEvent(FILE *fp);

    void skipNEvents(int n_to_skip);

    void close();
    void CloseWrite(int n_written);  //override nevents in the header and close
  };

  void print(std::string pfx, int itrack, const Track &trk, const Event &ev);

}  // end namespace mkfit
#endif
