#ifndef L1Trigger_TrackFindingTracklet_interface_TrackletCalculatorBase_h
#define L1Trigger_TrackFindingTracklet_interface_TrackletCalculatorBase_h

#include "L1Trigger/TrackFindingTracklet/interface/ProcessBase.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletParametersMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletProjectionsMemory.h"

#include <vector>

namespace trklet {

  class Settings;
  class Globals;
  class Stub;
  class L1TStub;
  class Tracklet;

  class TrackletCalculatorBase : public ProcessBase {
  public:
    TrackletCalculatorBase(std::string name, Settings const& settings, Globals* global);

    ~TrackletCalculatorBase() override = default;

    void exacttracklet(double r1,
                       double z1,
                       double phi1,
                       double r2,
                       double z2,
                       double phi2,
                       double,
                       double& rinv,
                       double& phi0,
                       double& t,
                       double& z0,
                       double phiproj[N_LAYER - 2],  //=4
                       double zproj[N_LAYER - 2],
                       double phider[N_LAYER - 2],
                       double zder[N_LAYER - 2],
                       double phiprojdisk[N_DISK],  //=5
                       double rprojdisk[N_DISK],
                       double phiderdisk[N_DISK],
                       double rderdisk[N_DISK]);

    void exacttrackletdisk(double r1,
                           double z1,
                           double phi1,
                           double r2,
                           double z2,
                           double phi2,
                           double,
                           double& rinv,
                           double& phi0,
                           double& t,
                           double& z0,
                           double phiprojLayer[N_PSLAYER],  //=3
                           double zprojLayer[N_PSLAYER],
                           double phiderLayer[N_PSLAYER],
                           double zderLayer[N_PSLAYER],
                           double phiproj[N_DISK - 2],  //=3
                           double rproj[N_DISK - 2],
                           double phider[N_DISK - 2],
                           double rder[N_DISK - 2]);

    void exacttrackletOverlap(double r1,
                              double z1,
                              double phi1,
                              double r2,
                              double z2,
                              double phi2,
                              double,
                              double& rinv,
                              double& phi0,
                              double& t,
                              double& z0,
                              double phiprojLayer[N_PSLAYER],  //=3
                              double zprojLayer[N_PSLAYER],
                              double phiderLayer[N_PSLAYER],
                              double zderLayer[N_PSLAYER],
                              double phiproj[N_DISK - 2],  //=3
                              double rproj[N_DISK - 2],
                              double phider[N_DISK - 2],
                              double rder[N_DISK - 2]);

    void exactproj(double rproj,
                   double rinv,
                   double phi0,
                   double t,
                   double z0,
                   double& phiproj,
                   double& zproj,
                   double& phider,
                   double& zder);

    void exactprojdisk(double zproj,
                       double rinv,
                       double phi0,
                       double t,
                       double z0,
                       double& phiproj,
                       double& rproj,
                       double& phider,
                       double& rder);

    void addDiskProj(Tracklet* tracklet, int disk);
    bool addLayerProj(Tracklet* tracklet, int layer);

    void addProjection(int layer, int iphi, TrackletProjectionsMemory* trackletprojs, Tracklet* tracklet);
    void addProjectionDisk(int disk, int iphi, TrackletProjectionsMemory* trackletprojs, Tracklet* tracklet);

    bool goodTrackPars(bool goodrinv, bool goodz0);

    bool inSector(int iphi0, int irinv, double phi0approx, double rinvapprox);

    bool barrelSeeding(const Stub* innerFPGAStub,
                       const L1TStub* innerStub,
                       const Stub* outerFPGAStub,
                       const L1TStub* outerStub);
    bool diskSeeding(const Stub* innerFPGAStub,
                     const L1TStub* innerStub,
                     const Stub* outerFPGAStub,
                     const L1TStub* outerStub);
    bool overlapSeeding(const Stub* innerFPGAStub,
                        const L1TStub* innerStub,
                        const Stub* outerFPGAStub,
                        const L1TStub* outerStub);

  protected:
    unsigned int iSeed_;
    unsigned int layerdisk1_;
    unsigned int layerdisk2_;

    int TCIndex_;

    unsigned int iSector_;
    double phimin_, phimax_;

    TrackletParametersMemory* trackletpars_;

    //First index is layer/disk second is phi region
    std::vector<std::vector<TrackletProjectionsMemory*> > trackletprojlayers_;
    std::vector<std::vector<TrackletProjectionsMemory*> > trackletprojdisks_;
  };

};  // namespace trklet
#endif
