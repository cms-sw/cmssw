//This class holds functional blocks of a sector
#ifndef L1Trigger_TrackFindingTracklet_interface_Sector_h
#define L1Trigger_TrackFindingTracklet_interface_Sector_h

#include "L1Trigger/TrackFindingTracklet/interface/L1TStub.h"
#include "L1Trigger/TrackFindingTracklet/interface/SLHCEvent.h"

#include <string>
#include <map>
#include <memory>
#include <vector>
#include <unordered_set>
#include <fstream>

namespace trklet {

  class Settings;
  class Globals;
  class ProcessBase;
  class MemoryBase;
  class Tracklet;
  class Track;
  class Stub;

  //Memory modules
  class InputLinkMemory;
  class AllStubsMemory;
  class VMStubsTEMemory;
  class VMStubsMEMemory;
  class StubPairsMemory;
  class StubTripletsMemory;
  class TrackletParametersMemory;
  class TrackletProjectionsMemory;
  class AllProjectionsMemory;
  class VMProjectionsMemory;
  class CandidateMatchMemory;
  class FullMatchMemory;
  class TrackFitMemory;
  class CleanTrackMemory;

  //Processing modules
  class VMRouter;
  class TrackletEngine;
  class TrackletEngineDisplaced;
  class TripletEngine;
  class TrackletCalculator;
  class TrackletProcessor;
  class TrackletCalculatorDisplaced;
  class ProjectionRouter;
  class MatchEngine;
  class MatchCalculator;
  class MatchProcessor;
  class FitTrack;
  class PurgeDuplicate;

  class Sector {
  public:
    Sector(unsigned int i, const Settings* settings, Globals* globals);

    ~Sector();

    bool addStub(L1TStub stub, std::string dtc);  //TODO - should be pointer or string

    // Creates all required memory modules based on wiring map (args: module type, module instance)
    void addMem(std::string memType, std::string memName);

    // Creates all required processing modules based on wiring map (args: module type, module instance)
    void addProc(std::string procType, std::string procName);

    //--- Create all required proc -> mem module connections, based on wiring map
    //--- (args: memory instance & input/output proc modules it connects to in format procName.pinName)
    void addWire(std::string mem, std::string procinfull, std::string procoutfull);

    ProcessBase* getProc(std::string procName);
    MemoryBase* getMem(std::string memName);

    void writeInputStubs(bool first);
    void writeVMSTE(bool first);
    void writeVMSME(bool first);
    void writeAS(bool first);
    void writeSP(bool first);
    void writeST(bool first);
    void writeTPAR(bool first);
    void writeTPROJ(bool first);
    void writeAP(bool first);
    void writeVMPROJ(bool first);
    void writeCM(bool first);
    void writeMC(bool first);
    void writeTF(bool first);
    void writeCT(bool first);

    void clean();

    // execute the different tracklet processing modules
    void executeVMR();
    void executeTE();
    void executeTED();
    void executeTRE();
    void executeTP();
    void executeTC();
    void executeTCD();
    void executePR();
    void executeME();
    void executeMC();
    void executeMP();
    void executeFT();
    void executePD(std::vector<Track*>& tracks);

    std::vector<Tracklet*> getAllTracklets() const;
    std::vector<const Stub*> getStubs() const;

    std::unordered_set<int> seedMatch(int itp) const;

    double phimin() const { return phimin_; }
    double phimax() const { return phimax_; }

    template <typename TV>
    void addMemToVec(std::vector<TV*>& memvec, TV* mem, const std::string& memName) {
      memvec.push_back(mem);
      Memories_[memName].reset(mem);
      MemoriesV_.push_back(mem);
    }

    template <typename TV>
    void addProcToVec(std::vector<TV*>& procvec, TV* proc, const std::string& procName) {
      procvec.push_back(proc);
      Processes_[procName].reset(proc);
    }

  private:
    int isector_;
    const Settings* const settings_;
    Globals* globals_;
    double phimin_;
    double phimax_;

    std::map<std::string, std::unique_ptr<MemoryBase> > Memories_;
    std::vector<MemoryBase*> MemoriesV_;
    std::vector<InputLinkMemory*> IL_;
    std::vector<AllStubsMemory*> AS_;
    std::vector<VMStubsTEMemory*> VMSTE_;
    std::vector<VMStubsMEMemory*> VMSME_;
    std::vector<StubPairsMemory*> SP_;
    std::vector<StubTripletsMemory*> ST_;
    std::vector<TrackletParametersMemory*> TPAR_;
    std::vector<TrackletProjectionsMemory*> TPROJ_;
    std::vector<AllProjectionsMemory*> AP_;
    std::vector<VMProjectionsMemory*> VMPROJ_;
    std::vector<CandidateMatchMemory*> CM_;
    std::vector<FullMatchMemory*> FM_;
    std::vector<TrackFitMemory*> TF_;
    std::vector<CleanTrackMemory*> CT_;

    std::map<std::string, std::unique_ptr<ProcessBase> > Processes_;
    std::vector<VMRouter*> VMR_;
    std::vector<TrackletEngine*> TE_;
    std::vector<TrackletEngineDisplaced*> TED_;
    std::vector<TripletEngine*> TRE_;
    std::vector<TrackletProcessor*> TP_;
    std::vector<TrackletCalculator*> TC_;
    std::vector<TrackletCalculatorDisplaced*> TCD_;
    std::vector<ProjectionRouter*> PR_;
    std::vector<MatchEngine*> ME_;
    std::vector<MatchCalculator*> MC_;
    std::vector<MatchProcessor*> MP_;
    std::vector<FitTrack*> FT_;
    std::vector<PurgeDuplicate*> PD_;
  };
};  // namespace trklet
#endif
