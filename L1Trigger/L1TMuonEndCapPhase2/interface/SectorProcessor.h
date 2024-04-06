#ifndef L1Trigger_L1TMuonEndCapPhase2_SectorProcessor_h
#define L1Trigger_L1TMuonEndCapPhase2_SectorProcessor_h

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"

namespace emtf::phase2 {

  class SectorProcessor {
  public:
    SectorProcessor(const EMTFContext&, const int&, const int&);

    ~SectorProcessor();

    void configureEvent(const edm::Event&);

    void configureBx(const int&);

    void select(const TriggerPrimitive&, const TPInfo&);

    void process(EMTFHitCollection&, EMTFTrackCollection&, EMTFInputCollection&);

  private:
    const EMTFContext& context_;

    int endcap_, sector_;
    std::map<SubsystemType, std::unique_ptr<TPSelector>> tp_selectors_;
    std::map<SubsystemType, std::unique_ptr<TPConverter>> tp_converters_;

    // Event
    const edm::Event* event_;
    const int* bx_;

    // Buffers
    std::vector<EMTFHitCollection> bx_window_hits_;
    std::map<SubsystemType, ILinkTPCMap> bx_ilink_tpc_maps_;

    // Helper functions
    void copyTP(const ILinkTPCMap& source, ILinkTPCMap& target) const;

    void convertTP(const int&, const ILinkTPCMap&, EMTFHitCollection&);

    void populateSegments(const std::vector<EMTFHitCollection>&, std::map<int, int>&, segment_collection_t&);

    void buildTracks(const std::map<int, int>&, const segment_collection_t&, const bool&, EMTFTrackCollection&);
  };

}  // namespace emtf::phase2

#endif  // L1Trigger_L1TMuonEndCapPhase2_SectorProcessor_h
