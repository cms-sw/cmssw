/****************************************************************
 * MatchEngineUnit (MEU) is a single instance of the MatchEngine
 * section of the MatchProcessor (MP)
 * 
 * Manual pipelining is implemented to properly emulate the HLS
 * implementation (required to meet II=1)
 * 
 * A total of `nMatchEngines_` MEUs are used in the MP
 ****************************************************************/
#ifndef L1Trigger_TrackFindingTracklet_interface_MatchEngineUnit_h
#define L1Trigger_TrackFindingTracklet_interface_MatchEngineUnit_h

#include "L1Trigger/TrackFindingTracklet/interface/VMStubsMEMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/CircularBuffer.h"
#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"
#include "L1Trigger/TrackFindingTracklet/interface/MemoryBase.h"

#include <cassert>
#include <vector>

namespace trklet {

  class Settings;
  class Stub;
  class L1TStub;
  class TrackletLUT;

  class MatchEngineUnit {
  public:
    MatchEngineUnit(const Settings& settings, bool barrel, unsigned int layerdisk, const TrackletLUT& luttable);

    ~MatchEngineUnit() = default;

    void init(VMStubsMEMemory* vmstubsmemory,
              unsigned int nrzbin,
              unsigned int rzbin,
              unsigned int iphi,
              int shift,
              int projrinv,
              int projfinerz,
              int projfinephi,
              bool usefirstMinus,
              bool usefirstPlus,
              bool usesecondMinus,
              bool usesecondPlus,
              bool isPSseed,
              Tracklet* proj);

    bool empty() const { return candmatches_.empty(); }

    int TCID() const;

    std::pair<Tracklet*, const Stub*> read() { return candmatches_.read(); }

    std::pair<Tracklet*, const Stub*> peek() const { return candmatches_.peek(); }

    bool idle() const { return idle_; }

    bool active() const { return !idle_ || good_in || good_out || !empty(); }

    void setAlmostFull();

    void setimeu(int imeu) { imeu_ = imeu; }

    void setprint(bool print) { print_ = print; }

    void reset();

    unsigned int rptr() const { return candmatches_.rptr(); }
    unsigned int wptr() const { return candmatches_.wptr(); }

    void step();

    void processPipeline();

  private:
    //Provide access to constants
    const Settings& settings_;

    VMStubsMEMemory* vmstubsmemory_;

    unsigned int nrzbins_;
    unsigned int rzbin_, rzbin_in, rzbin_out, rzbin_pipeline;
    unsigned int phibin_;
    int shift_;

    unsigned int istub_;
    unsigned int iuse_;

    bool barrel_;
    int projrinv_;
    int projfinerz_;
    int projfinephi_;
    std::vector<std::pair<unsigned int, unsigned int>> use_;
    bool isPSseed_;
    Tracklet* proj_;

    bool idle_;

    unsigned int layerdisk_;

    //The minimum radius for 2s disks in projection bins
    unsigned int ir2smin_;

    //Save state at the start of istep
    bool almostfullsave_;

    //LUT for bend consistency with rinv
    const TrackletLUT& luttable_;

    //Various manually pipelined variables
    //Each _ represents a layer of pipelining
    //e.g., good_in is set and one iteration later good_out is updated
    VMStubME vmstub_in, vmstub_pipeline, vmstub_out;
    bool isPSseed_in, isPSseed_pipeline, isPSseed_out;
    bool good_in, good_pipeline, good_out;
    int projfinerz_in, projfinerz_pipeline, projfinerz_out;
    int projfinephi_in, projfinephi_pipeline, projfinephi_out;
    int projrinv_in, projrinv_pipeline, projrinv_out;
    Tracklet *proj_in, *proj_pipeline, *proj_out;

    //save the candidate matches
    CircularBuffer<std::pair<Tracklet*, const Stub*>> candmatches_;

    //debugging help
    int imeu_;
    bool print_;
  };

};  // namespace trklet
#endif
