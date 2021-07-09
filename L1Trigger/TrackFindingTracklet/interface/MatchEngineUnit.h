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

    bool active() const { return !idle_ || good__ || good___ || !empty(); }

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
    unsigned int rzbin_;
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

    //Save state at the start of istep
    bool almostfullsave_;

    //LUT for bend consistency with rinv
    const TrackletLUT& luttable_;

    //Various manually pipelined variables
    //Each _ represents a layer of pipelining
    //e.g., good__ is set and one iteration later good___ is updated
    VMStubME vmstub__, vmstub___;
    bool isPSseed__, isPSseed___;
    bool good__, good___;
    int projfinerz__, projfinerz___;
    int projfinephi__, projfinephi___;
    int projrinv__, projrinv___;
    Tracklet *proj__, *proj___;

    //save the candidate matches
    CircularBuffer<std::pair<Tracklet*, const Stub*>> candmatches_;

    //debugging help
    int imeu_;
    bool print_;
  };

};  // namespace trklet
#endif
