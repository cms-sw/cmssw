#ifndef L1Trigger_TrackFindingTracklet_interface_TrackletCalculatorDisplaced_h
#define L1Trigger_TrackFindingTracklet_interface_TrackletCalculatorDisplaced_h

#include "L1Trigger/TrackFindingTracklet/interface/ProcessBase.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletProjectionsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/StubTripletsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/AllStubsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletParametersMemory.h"

#include <vector>

namespace trklet {

  class Settings;
  class Globals;
  class MemoryBase;
  class Stub;
  class L1TStub;

  class TrackletCalculatorDisplaced : public ProcessBase {
  public:
    TrackletCalculatorDisplaced(std::string name, Settings const& settings, Globals* global);

    ~TrackletCalculatorDisplaced() override = default;

    void addOutputProjection(TrackletProjectionsMemory*& outputProj, MemoryBase* memory);

    void addOutput(MemoryBase* memory, std::string output) override;
    void addInput(MemoryBase* memory, std::string input) override;

    void execute(unsigned int iSector, double phimin, double phimax);

    void addDiskProj(Tracklet* tracklet, int disk);
    bool addLayerProj(Tracklet* tracklet, int layer);

    void addProjection(int layer, int iphi, TrackletProjectionsMemory* trackletprojs, Tracklet* tracklet);
    void addProjectionDisk(int disk, int iphi, TrackletProjectionsMemory* trackletprojs, Tracklet* tracklet);

    bool LLLSeeding(const Stub* innerFPGAStub,
                    const L1TStub* innerStub,
                    const Stub* middleFPGAStub,
                    const L1TStub* middleStub,
                    const Stub* outerFPGAStub,
                    const L1TStub* outerStub);
    bool DDLSeeding(const Stub* innerFPGAStub,
                    const L1TStub* innerStub,
                    const Stub* middleFPGAStub,
                    const L1TStub* middleStub,
                    const Stub* outerFPGAStub,
                    const L1TStub* outerStub);
    bool LLDSeeding(const Stub* innerFPGAStub,
                    const L1TStub* innerStub,
                    const Stub* middleFPGAStub,
                    const L1TStub* middleStub,
                    const Stub* outerFPGAStub,
                    const L1TStub* outerStub);

    void exactproj(double rproj,
                   double rinv,
                   double phi0,
                   double d0,
                   double t,
                   double z0,
                   double r0,
                   double& phiproj,
                   double& zproj,
                   double& phider,
                   double& zder);

    void exactprojdisk(double zproj,
                       double rinv,
                       double,
                       double,  //phi0 and d0 are not used.
                       double t,
                       double z0,
                       double x0,
                       double y0,
                       double& phiproj,
                       double& rproj,
                       double& phider,
                       double& rder);

    void exacttracklet(double r1,
                       double z1,
                       double phi1,
                       double r2,
                       double z2,
                       double phi2,
                       double r3,
                       double z3,
                       double phi3,
                       int take3,
                       double& rinv,
                       double& phi0,
                       double& d0,
                       double& t,
                       double& z0,
                       double phiproj[N_LAYER - 2],
                       double zproj[N_LAYER - 2],
                       double phiprojdisk[N_DISK],
                       double rprojdisk[N_DISK],
                       double phider[N_LAYER - 2],
                       double zder[N_LAYER - 2],
                       double phiderdisk[N_DISK],
                       double rderdisk[N_DISK]);

    void approxproj(double halfRinv,
                    double phi0,
                    double d0,
                    double t,
                    double z0,
                    double halfRinv_0,
                    double d0_0,  // zeroeth order result for higher order terms calculation
                    double rmean,
                    double& phiproj,
                    double& phiprojder,
                    double& zproj,
                    double& zprojder);

    void approxprojdisk(double halfRinv,
                        double phi0,
                        double d0,
                        double t,
                        double z0,
                        double halfRinv_0,
                        double d0_0,  // zeroeth order result for higher order terms calculation
                        double zmean,
                        double& phiproj,
                        double& phiprojder,
                        double& rproj,
                        double& rprojder);

    void approxtracklet(double r1,
                        double z1,
                        double phi1,
                        double r2,
                        double z2,
                        double phi2,
                        double r3,
                        double z3,
                        double phi3,
                        bool take3,
                        unsigned ndisks,
                        double& rinv,
                        double& phi0,
                        double& d0,
                        double& t,
                        double& z0,
                        double phiproj[4],
                        double zproj[4],
                        double phider[4],
                        double zder[4],
                        double phiprojdisk[5],
                        double rprojdisk[5],
                        double phiderdisk[5],
                        double rderdisk[5]);

  protected:
    int TCIndex_;
    int layer_;
    int disk_;
    unsigned int iSeed_;
    double rproj_[N_LAYER - 2];
    int lproj_[N_LAYER - 2];
    double zproj_[N_DISK - 2];
    int dproj_[N_DISK - 2];
    double rzmeanInv_[N_DISK - 2];

    unsigned int iSector_;
    double phimin_, phimax_;

    std::vector<double> toR_;
    std::vector<double> toZ_;

    std::vector<AllStubsMemory*> innerallstubs_;
    std::vector<AllStubsMemory*> middleallstubs_;
    std::vector<AllStubsMemory*> outerallstubs_;
    std::vector<StubTripletsMemory*> stubtriplets_;

    TrackletParametersMemory* trackletpars_;

    //First index is layer/disk second is phi region
    std::vector<std::vector<TrackletProjectionsMemory*> > trackletprojlayers_;
    std::vector<std::vector<TrackletProjectionsMemory*> > trackletprojdisks_;
  };

};  // namespace trklet
#endif
