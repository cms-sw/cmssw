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

    void init(int iSeed);

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


    void projlayer(int ir, int irinv, int iphi0, int it,int iz0, int &iz, int &iphi); // lower case to differentiate from ProjectionCalculator functions

    void projdisk(int iz, int irinv, int iphi0, int it,int iz0, int &ir, int &iphi, int &iderphi, int &iderr);

    void calcPars(unsigned int idr, int iphi1, int ir1, int iz1, int iphi2, int ir2, int iz2,
		  int &irinv_new, int &iphi0_new, int &iz0_new, int &it_new);

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
                     const L1TStub* outerStub,
		     bool print =  false);
    bool overlapSeeding(const Stub* innerFPGAStub,
                        const L1TStub* innerStub,
                        const Stub* outerFPGAStub,
                        const L1TStub* outerStub,
			bool print =  false);


  protected:
    unsigned int iSeed_;
    unsigned int layerdisk1_;
    unsigned int layerdisk2_;

    int TCIndex_;

    unsigned int iSector_;
    double phimin_, phimax_;
    double phiHG_;



    TrackletParametersMemory* trackletpars_;

    //First index is layer/disk second is phi region
    std::vector<std::vector<TrackletProjectionsMemory*> > trackletprojlayers_;
    std::vector<std::vector<TrackletProjectionsMemory*> > trackletprojdisks_;

    //Constants for coordinates and track parameter definitions
    int n_phi_;
    int n_r_;
    int n_z_;
    int n_phi0_;
    int n_rinv_;
    int n_t_;
    int n_phidisk_;
    int n_rdisk_;
  
    //Constants used for tracklet parameter calculations
    int n_Deltar_;
    int n_delta0_;
    int n_deltaz_;
    int n_delta1_;
    int n_delta2_;
    int n_delta12_;
    int n_a_;
    int n_r6_;
    int n_delta02_;
    int n_x6_;
    int n_HG_;  

    //Constants used for projectison to layers
    int n_s_;
    int n_s6_;

    //Constants used for projectison to disks
    int n_tinv_;
    int n_y_;
    int n_x_;
    int n_xx6_;
  
    std::vector<int> LUT_itinv_;
    std::vector<int> LUT_idrinv_;


  };

};  // namespace trklet
#endif
