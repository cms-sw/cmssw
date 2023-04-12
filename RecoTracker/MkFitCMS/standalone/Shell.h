#ifndef RecoTracker_MkFitCMS_standalone_Shell_h
#define RecoTracker_MkFitCMS_standalone_Shell_h

#include "RecoTracker/MkFitCore/interface/Hit.h"
#include "RecoTracker/MkFitCore/interface/Track.h"

#include <map>

namespace mkfit {

  class DataFile;
  class Event;
  class EventOfHits;
  class MkBuilder;

  class Shell {
  public:
    enum SeedSelect_e { SS_UseAll = 0, SS_Label, SS_IndexPreCleaning, SS_IndexPostCleaning };

    Shell(std::vector<DeadVec> &dv, const std::string &in_file, int start_ev);
    void Run();

    void Status();

    void GoToEvent(int eid);
    void NextEvent(int skip = 1);
    void ProcessEvent(SeedSelect_e seed_select = SS_UseAll, int selected_seed = -1, int count = 1);

    void SelectIterationIndex(int itidx);
    void SelectIterationAlgo(int algo);
    void PrintIterations();

    void SetDebug(bool b);
    void SetCleanSeeds(bool b);
    void SetBackwardFit(bool b);
    void SetRemoveDuplicates(bool b);
    void SetUseDeadModules(bool b);

    Event *event() { return m_event; }
    EventOfHits *eoh() { return m_eoh; }
    MkBuilder *builder() { return m_builder; }

    // --------------------------------------------------------
    // Analysis helpers

    int LabelFromHits(Track &t, bool replace, float good_frac);
    void FillByLabelMaps_CkfBase();

    bool CheckMkFitLayerPlanVsReferenceHits(const Track &mkft, const Track &reft, const std::string &name);

    // --------------------------------------------------------
    // Analysis drivers / main functions / Comparators

    void Compare();

  private:
    std::vector<DeadVec> &m_deadvectors;
    DataFile *m_data_file = nullptr;
    Event *m_event = nullptr;
    EventOfHits *m_eoh = nullptr;
    MkBuilder *m_builder = nullptr;
    int m_evs_in_file = -1;
    int m_it_index = 0;
    bool m_clean_seeds = true;
    bool m_backward_fit = true;
    bool m_remove_duplicates = true;

    TrackVec m_seeds;
    TrackVec m_tracks;

    using map_t = std::map<int, Track *>;
    using map_i = map_t::iterator;

    std::map<int, Track *> m_ckf_map, m_sim_map, m_seed_map, m_mkf_map;
  };

}  // namespace mkfit

#endif
