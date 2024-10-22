#ifndef RecoTracker_MkFitCore_interface_IdxChi2List_h
#define RecoTracker_MkFitCore_interface_IdxChi2List_h

namespace mkfit {

  struct IdxChi2List {
  public:
    unsigned int module;  // module id
    int hitIdx;           // hit index
    int trkIdx;           // candidate index
    int nhits;            // number of hits (used for sorting)
    int ntailholes;       // number of holes at the end of the track (used for sorting)
    int noverlaps;        // number of overlaps (used for sorting)
    int nholes;           // number of holes (used for sorting)
    float pt;             // pt (used for sorting)
    float chi2;           // total chi2 (used for sorting)
    float chi2_hit;       // chi2 of the added hit
    float score;          // score used for candidate ranking

    // Zero initialization
    void reset() {
      module = 0u;
      hitIdx = trkIdx = 0;
      nhits = ntailholes = noverlaps = nholes = 0;
      pt = chi2 = chi2_hit = score = 0;
    }
  };

}  // namespace mkfit

#endif
