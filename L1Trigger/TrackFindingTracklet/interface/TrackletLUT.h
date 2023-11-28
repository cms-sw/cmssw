//////////////////////////////////////////////////////////////////
// TrackletLUT
// This class writes out the variuos look up tables
// for all modules.
//////////////////////////////////////////////////////////////////
//
// This class has methods to build the LUT (LookUp Tables) used by the track finding
// It also provides a method to write out the file for use by the firmware implementation
//
//
#ifndef L1Trigger_TrackFindingTracklet_interface_TrackletLUT_h
#define L1Trigger_TrackFindingTracklet_interface_TrackletLUT_h

#include <string>
#include <vector>
#include <string>

#include "L1Trigger/TrackTrigger/interface/Setup.h"

class Setup;

namespace trklet {

  class Settings;

  class TrackletLUT {
  public:
    TrackletLUT(const Settings& settings);

    TrackletLUT& operator=(const TrackletLUT& other) {
      name_ = other.name_;
      table_ = other.table_;
      nbits_ = other.nbits_;
      positive_ = other.positive_;

      return *this;
    }

    ~TrackletLUT() = default;

    enum MatchType {
      barrelphi,   // write LUT for barrel phi
      barrelz,     // write LUT for barrel z
      disk2Sphi,   // write LUT for disk 2S phi
      disk2Sr,     // write LUT for disk 2S r
      diskPSphi,   // write LUT for disk PS phi
      diskPSr,     // write LUT for disk PS r
      alphainner,  // write alpha corrections LUT for 2S (inner)
      alphaouter,  // write alpha corrections LUT for 2S (outer)
      rSSinner,    // write r LUT for 2S (inner)
      rSSouter     // write r LUT for 2S (inner)
    };

    //region only used for name - should be removed
    void initmatchcut(unsigned int layerdisk, MatchType type, unsigned int region);

    //Builds LUT that for each TP returns if the phi differences between inner and outer
    //stub is consistent with the pT cut and the stub pair should be kept.
    void initTPlut(bool fillInner,
                   unsigned int iSeed,
                   unsigned int layerdisk1,
                   unsigned int layerdisk2,
                   unsigned int nbitsfinephidiff,
                   unsigned int iTP);

    //Builds a lut for the TP ro decide if the region should be used. This is used in the
    //first stage of the TP to decide which regions in the outer layer an inner stub needs
    //to be combined with
    void initTPregionlut(unsigned int iSeed,
                         unsigned int layerdisk1,
                         unsigned int layerdisk2,
                         unsigned int iAllStub,
                         unsigned int nbitsfinephidiff,
                         unsigned int nbitsfinephi,
                         const TrackletLUT& tplutinner,
                         unsigned int iTP);

    //Stub pt consistency for tracklet engine
    void initteptlut(bool fillInner,
                     bool fillTEMem,
                     unsigned int iSeed,
                     unsigned int layerdisk1,
                     unsigned int layerdisk2,
                     unsigned int innerphibits,
                     unsigned int outerphibits,
                     double innerphimin,
                     double innerphimax,
                     double outerphimin,
                     double outerphimax,
                     const std::string& innermem,
                     const std::string& outermem);

    //This LUT uses the phi derivative known in the projection to calculate the corresponding bend
    void initProjectionBend(double k_phider, unsigned int idisk, unsigned int nrbits, unsigned int nphiderbits);

    //This LUT implements consistence check for match engine to check that stub bend is consistent with projection
    void initBendMatch(unsigned int layerdisk);

    void initProjectionDiskRadius(int nrbits);

    enum VMRTableType { me, disk, inner, inneroverlap, innerthird };

    //In the VMR we used the position of the stub (r, z) to calculate the bin and fine rz position the stub has
    //region only used for name - should be removed
    void initVMRTable(unsigned int layerdisk, VMRTableType type, int region = -1);

    //Used in barrel to calculate the phi position of a stub at the nominal radis of the layer based on the stub radial
    //psotion and bend
    void initPhiCorrTable(unsigned int layerdisk, unsigned int rbits);

    //writes out the LUT in standared format for firmware
    void writeTable() const;

    //Evaluate the LUT
    int lookup(unsigned int index) const;

    unsigned int size() const { return table_.size(); }

  private:
    const Settings& settings_;
    const tt::Setup* setup_;

    //Determine bend/bend cuts in LUT regions
    std::vector<const tt::SensorModule*> getSensorModules(unsigned int layerdisk,
                                                          bool isPS,
                                                          std::array<double, 2> tan_range = {{-1, -1}},
                                                          unsigned int nzbins = 1,
                                                          unsigned int zbin = 0);

    std::array<double, 2> getTanRange(const std::vector<const tt::SensorModule*>& sensorModules);

    std::vector<std::array<double, 2>> getBendCut(unsigned int layerdisk,
                                                  const std::vector<const tt::SensorModule*>& sensorModules,
                                                  bool isPS,
                                                  double FEbendcut = 0);

    int getphiCorrValue(
        unsigned int layerdisk, double bend, unsigned int irbin, double rmean, double dr, double drmax) const;

    int getVMRLookup(unsigned int layerdisk, double z, double r, double dz, double dr, int iseed = -1) const;

    std::string name_;

    std::vector<int> table_;

    unsigned int nbits_;

    bool positive_;
  };
};  // namespace trklet
#endif
