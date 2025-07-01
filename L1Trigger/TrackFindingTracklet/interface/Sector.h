//This class holds functional blocks of a sector
#ifndef L1Trigger_TrackFindingTracklet_interface_Sector_h
#define L1Trigger_TrackFindingTracklet_interface_Sector_h

#include "L1Trigger/TrackFindingTracklet/interface/L1TStub.h"
#include "L1Trigger/TrackFindingTracklet/interface/SLHCEvent.h"
#include "L1Trigger/TrackFindingTracklet/interface/StubStreamData.h"

#include <string>
#include <map>
#include <memory>
#include <vector>
#include <unordered_set>
#include <fstream>
#include <unordered_map>

namespace trklet {

  class Settings;
  class Globals;
  class ProcessBase;
  class MemoryBase;
  class Tracklet;
  class Track;
  class Stub;

  //Memory modules
  class DTCLinkMemory;
  class InputLinkMemory;
  class AllStubsMemory;
  class AllInnerStubsMemory;
  class VMStubsTEMemory;
  class VMStubsMEMemory;
  class StubTripletsMemory;
  class TrackletParametersMemory;
  class TrackletProjectionsMemory;
  class FullMatchMemory;
  class TrackFitMemory;
  class CleanTrackMemory;

  //Processing modules
  class InputRouter;
  class VMRouterCM;
  class TrackletProcessor;
  class TrackletProcessorDisplaced;
  class TrackletCalculatorDisplaced;
  class ProjectionCalculator;
  class VMStubMERouter;
  class MatchProcessor;
  class FitTrack;
  class PurgeDuplicate;

  class Sector {
  public:
    Sector(Settings const& settings, Globals* globals);

    ~Sector();

    //Set the sector
    void setSector(unsigned int isector);

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

    void writeDTCStubs(bool first);
    void writeIRStubs(bool first);
    void writeVMSTE(bool first);
    void writeVMSME(bool first);
    void writeAS(bool first);
    void writeAIS(bool first);
    void writeST(bool first);
    void writeTPAR(bool first);
    void writeTPROJ(bool first);
    void writeMC(bool first);
    void writeTF(bool first);
    void writeCT(bool first);

    void clean();

    // execute the different tracklet processing modules
    void executeIR();
    void executeVMR();
    void executeTED();
    void executeTRE();
    void executeTP();
    void executeTPD();
    void executeTCD();
    void executePC();
    void executeVMSMER();
    void executeMP();
    void executeFT(std::vector<std::vector<std::string>>& streamsTrackRaw,
                   std::vector<std::vector<StubStreamData>>& streamsStubRaw);
    void executePD(std::vector<Track>& tracks);

    std::vector<Tracklet*> getAllTracklets() const;
    std::vector<const Stub*> getStubs() const;

    std::unordered_set<int> seedMatch(int itp) const;

    double phimin() const { return phimin_; }
    double phimax() const { return phimax_; }

    template <typename TV, typename... Args>
    void addMemToVec(std::vector<std::unique_ptr<TV>>& memvec, const std::string& memName, Args&... args) {
      memvec.push_back(std::make_unique<TV>(memName, std::forward<Args>(args)...));
      Memories_[memName] = memvec.back().get();
      MemoriesV_.push_back(memvec.back().get());
    }

    template <typename TV, typename... Args>
    void addProcToVec(std::vector<std::unique_ptr<TV>>& procvec, const std::string& procName, Args&... args) {
      procvec.push_back(std::make_unique<TV>(procName, std::forward<Args>(args)...));
      Processes_[procName] = procvec.back().get();
    }

  private:
    int isector_;
    Settings const& settings_;
    Globals* globals_;
    double phimin_;
    double phimax_;

    std::map<std::string, MemoryBase*> Memories_;
    std::vector<MemoryBase*> MemoriesV_;
    std::vector<std::unique_ptr<DTCLinkMemory>> DL_;
    std::vector<std::unique_ptr<InputLinkMemory>> IL_;
    std::vector<std::unique_ptr<AllStubsMemory>> AS_;
    std::vector<std::unique_ptr<AllInnerStubsMemory>> AIS_;
    std::vector<std::unique_ptr<VMStubsTEMemory>> VMSTE_;
    std::vector<std::unique_ptr<VMStubsMEMemory>> VMSME_;
    std::vector<std::unique_ptr<StubTripletsMemory>> ST_;
    std::vector<std::unique_ptr<TrackletParametersMemory>> TPAR_;
    std::vector<std::unique_ptr<TrackletProjectionsMemory>> TPROJ_;
    std::vector<std::unique_ptr<FullMatchMemory>> FM_;
    std::vector<std::unique_ptr<TrackFitMemory>> TF_;
    std::vector<std::unique_ptr<CleanTrackMemory>> CT_;

    std::map<std::string, ProcessBase*> Processes_;
    std::vector<std::unique_ptr<InputRouter>> IR_;
    std::vector<std::unique_ptr<VMRouterCM>> VMRCM_;
    std::vector<std::unique_ptr<TrackletProcessor>> TP_;
    std::vector<std::unique_ptr<TrackletProcessorDisplaced>> TPD_;
    std::vector<std::unique_ptr<TrackletCalculatorDisplaced>> TCD_;
    std::vector<std::unique_ptr<ProjectionCalculator>> PC_;
    std::vector<std::unique_ptr<VMStubMERouter>> VMSMER_;
    std::vector<std::unique_ptr<MatchProcessor>> MP_;
    std::vector<std::unique_ptr<FitTrack>> FT_;
    std::vector<std::unique_ptr<PurgeDuplicate>> PD_;
  };
};  // namespace trklet
#endif
