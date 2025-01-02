#ifndef L1Trigger_TrackFindingTracklet_interface_ProjectionCalculator_h
#define L1Trigger_TrackFindingTracklet_interface_ProjectionCalculator_h

#include "L1Trigger/TrackFindingTracklet/interface/ProcessBase.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletLUT.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletProjectionsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletParametersMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletConfigBuilder.h"

namespace trklet {

  class Settings;
  class Globals;
  class MemoryBase;

  class ProjectionCalculator : public ProcessBase {
  public:
    ProjectionCalculator(std::string name, Settings const& settings, Globals* global);

    ~ProjectionCalculator() override = default;

    void addOutput(MemoryBase* memory, std::string output) override;
    void addInput(MemoryBase* memory, std::string input) override;

    void execute();

    void projLayer(int ir, int irinv, int iphi0, int it,int iz0, int &iz, int &iphi);

    void projDisk(int iz, int irinv, int iphi0, int it,int iz0, int &ir, int &iphi, int &iderphi, int &iderr);

  private:

    std::vector<std::vector<std::vector<TrackletProjectionsMemory*> > > outputproj_; // projs now stored by layer/disk & phi region 

    std::vector<TrackletParametersMemory*> inputpars_;
    std::vector<TrackletParametersMemory*> outputpars_;
    std::vector<std::string > projnames_; 

    //Constants for coordinates and track parameter definitions
    int n_phi_;
    int n_r_;
    int n_z_;
    int n_phi0_;
    int n_rinv_;
    int n_t_;
    int n_phidisk_;
    int n_rdisk_;

    //Constants used for projectison to layers
    int n_s_;
    int n_s6_;

    //Constants used for projectison to disks
    int n_tinv_;
    int n_y_;
    int n_x_;
    int n_xx6_;

    unsigned int nMergedTC[8] = {6, 1, 2, 1, 1, 1, 2, 1};

    double phiHG_;

    std::vector<int> LUT_itinv_;

  };

};  // namespace trklet
#endif
