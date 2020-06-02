#ifndef L1Trigger_TrackFindingTracklet_interface_TrackDerTable_h
#define L1Trigger_TrackFindingTracklet_interface_TrackDerTable_h

#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <vector>
#include <string>

#include "L1Trigger/TrackFindingTracklet/interface/TrackDer.h"

namespace trklet {

  class Settings;
  class Globals;

  class TrackDerTable {
  public:
    TrackDerTable(Settings const& settings);

    ~TrackDerTable() = default;

    const TrackDer* getDerivatives(int index) const { return &derivatives_[index]; }

    const TrackDer* getDerivatives(unsigned int layermask,
                                   unsigned int diskmask,
                                   unsigned int alphaindex,
                                   unsigned int rinvindex) const;

    int getIndex(unsigned int layermask, unsigned int diskmask) const;

    void addEntry(unsigned int layermask, unsigned int diskmask, int multiplicity, int nrinv);

    void readPatternFile(std::string fileName);

    int getEntries() const { return nextLayerDiskValue_; }

    void fillTable();

    static void invert(double M[4][8], unsigned int n);

    static void invert(std::vector<std::vector<double> >& M, unsigned int n);

    static void calculateDerivatives(Settings const& settings,
                                     unsigned int nlayers,
                                     double r[N_LAYER],
                                     unsigned int ndisks,
                                     double z[N_DISK],
                                     double alpha[N_DISK],
                                     double t,
                                     double rinv,
                                     double D[N_FITPARAM][N_FITSTUB * 2],
                                     int iD[N_FITPARAM][N_FITSTUB * 2],
                                     double MinvDt[N_FITPARAM][N_FITSTUB * 2],
                                     int iMinvDt[N_FITPARAM][N_FITSTUB * 2],
                                     double sigma[N_FITSTUB * 2],
                                     double kfactor[N_FITSTUB * 2]);

    static double tpar(Settings const& settings, int diskmask, int layermask);

  private:
    Settings const& settings_;

    std::vector<int> LayerMem_;
    std::vector<int> DiskMem_;
    std::vector<int> LayerDiskMem_;

    unsigned int LayerMemBits_;
    unsigned int DiskMemBits_;
    unsigned int LayerDiskMemBits_;
    unsigned int alphaBits_;

    unsigned int Nlay_;
    unsigned int Ndisk_;

    std::vector<TrackDer> derivatives_;

    int nextLayerValue_;
    int nextDiskValue_;
    int nextLayerDiskValue_;
    int lastMultiplicity_;
  };

};  // namespace trklet
#endif
