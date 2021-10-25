#ifndef L1Trigger_TrackFindingTracklet_interface_TrackletLUT_h
#define L1Trigger_TrackFindingTracklet_interface_TrackletLUT_h

#include <string>
#include <vector>
#include <string>

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

    enum MatchType { barrelphi, barrelz, disk2Sphi, disk2Sr, diskPSphi, diskPSr };

    //region only used for name - should be removed
    void initmatchcut(unsigned int layerdisk, MatchType type, unsigned int region);

    void initTPlut(bool fillInner,
                   unsigned int iSeed,
                   unsigned int layerdisk1,
                   unsigned int layerdisk2,
                   unsigned int nbitsfinephidiff,
                   unsigned int iTP);

    void initTPregionlut(unsigned int iSeed,
                         unsigned int layerdisk1,
                         unsigned int layerdisk2,
                         unsigned int iAllStub,
                         unsigned int nbitsfinephidiff,
                         unsigned int nbitsfinephi,
                         const TrackletLUT& tplutinner,
                         unsigned int iTP);

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

    void initProjectionBend(double k_phider, unsigned int idisk, unsigned int nrbits, unsigned int nphiderbits);

    void initBendMatch(unsigned int layerdisk);

    enum VMRTableType { me, disk, inner, inneroverlap, innerthird };

    //region only used for name - should be removed
    void initVMRTable(unsigned int layerdisk, VMRTableType type, int region = -1);

    void initPhiCorrTable(unsigned int layerdisk, unsigned int rbits);

    void writeTable() const;

    int lookup(unsigned int index) const;

    unsigned int size() const { return table_.size(); }

  private:
    int getphiCorrValue(
        unsigned int layerdisk, unsigned int ibend, unsigned int irbin, double rmean, double dr, double drmax) const;

    int getVMRLookup(unsigned int layerdisk, double z, double r, double dz, double dr, int iseed = -1) const;

    const Settings& settings_;

    std::string name_;

    std::vector<int> table_;

    unsigned int nbits_;

    bool positive_;
  };
};  // namespace trklet
#endif
