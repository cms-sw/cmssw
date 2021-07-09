//
// Class to build the configuration for the tracklet based track finding
//
//
//
#ifndef L1Trigger_TrackFindingTracklet_interface_TrackletConfigBuilder_h
#define L1Trigger_TrackFindingTracklet_interface_TrackletConfigBuilder_h

#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"

#include <vector>
#include <list>
#include <utility>
#include <set>
#include <iostream>
#include <fstream>
#include <cstdlib>

namespace tt {
  class Setup;
}

namespace trklet {

  class TrackletConfigBuilder {
  public:
    //Builds the configuration for the tracklet based track finding
    TrackletConfigBuilder(const Settings& settings, const tt::Setup* setup = nullptr);

    //This method writes out the configuration as files
    void writeAll(std::ostream& wires, std::ostream& memories, std::ostream& modules);

    //
    // The next group of fcn formats a string to write out names of different
    // memories and processing modules
    //

    //Seed string, eg. L1L2
    std::string iSeedStr(unsigned int iSeed) const;

    //Return unsigned as string
    static std::string numStr(unsigned int i);

    //Return iTC as string - ie A, B, C, etc
    std::string iTCStr(unsigned int iTC) const;

    //The region string A, B, C etc for layers and disks; X, Y, Z etc for overlap
    std::string iRegStr(unsigned int iReg, unsigned int iSeed) const;

    //TC Name
    std::string TCName(unsigned int iSeed, unsigned int iTC) const;

    //Name of layer or disk, e.g. L1 or D1
    static std::string LayerName(unsigned int ilayer);

    //Tracklet projection name
    std::string TPROJName(unsigned int iSeed, unsigned int iTC, unsigned int ilayer, unsigned int ireg) const;

    //Projection router name
    std::string PRName(unsigned int ilayer, unsigned int ireg) const;

  private:
    //
    // Method to initialize the regions and VM in each layer
    //
    void initGeom();

    //
    // Builds the list of TE for each seeding combination
    //
    void buildTE();

    //
    // Builds the lists of TC for each seeding combination
    //
    void buildTC();

    //
    // Finds the projections needed for each seeding combination
    //
    void buildProjections();

#ifdef CMSSW_GIT_HASH
    // Calculate phi range of modules read by each DTC.
    void setDTCphirange(const tt::Setup* setup);

    // Write DTC phi ranges to dtcphirange.txt for stand-alone emulation.
    void writeDTCphirange() const;
#else
    // Set phi ranges after reading them from dtcphirange.txt (stand-alone emulation)
    void setDTCphirange(const tt::Setup* setup = nullptr);
#endif

    //
    // Helper function to determine if a pair of VM memories form valid TE
    //
    bool validTEPair(unsigned int iseed, unsigned int iTE1, unsigned int iTE2);

    //
    // Helper fcn. to get the layers/disks for a seed
    //
    std::pair<unsigned int, unsigned int> seedLayers(unsigned int iSeed);

    //
    // Helper fcn to get the radii of the two layers in a seed
    //
    std::pair<double, double> seedRadii(unsigned int iseed);

    //
    // Helper fcn to return the phi range of a projection of a tracklet from a TC
    //
    std::pair<double, double> seedPhiRange(double rproj, unsigned int iSeed, unsigned int iTC);

    //
    // Helper function to calculate the phi position of a seed at radius r that is formed
    // by two stubs at (r1,phi1) and (r2, phi2)
    //
    double phi(double r1, double phi1, double r2, double phi2, double r);

    //
    // Helper function to calculate rinv for two stubs at (r1,phi1) and (r2,phi2)
    //
    double rinv(double r1, double phi1, double r2, double phi2);

    //StubPair Name
    std::string SPName(unsigned int l1,
                       unsigned int ireg1,
                       unsigned int ivm1,
                       unsigned int l2,
                       unsigned int ireg2,
                       unsigned int ivm2,
                       unsigned int iseed) const;

    //StubPair displaced name
    std::string SPDName(unsigned int l1,
                        unsigned int ireg1,
                        unsigned int ivm1,
                        unsigned int l2,
                        unsigned int ireg2,
                        unsigned int ivm2,
                        unsigned int l3,
                        unsigned int ireg3,
                        unsigned int ivm3,
                        unsigned int iseed) const;

    //Stub Triplet name
    std::string STName(unsigned int l1,
                       unsigned int ireg1,
                       unsigned int l2,
                       unsigned int ireg2,
                       unsigned int l3,
                       unsigned int ireg3,
                       unsigned int iseed,
                       unsigned int count) const;

    //TrackletEngine name
    std::string TEName(unsigned int l1,
                       unsigned int ireg1,
                       unsigned int ivm1,
                       unsigned int l2,
                       unsigned int ireg2,
                       unsigned int ivm2,
                       unsigned int iseed) const;

    //Triplet engine name
    std::string TREName(unsigned int l1,
                        unsigned int ireg1,
                        unsigned int l2,
                        unsigned int ireg2,
                        unsigned int iseed,
                        unsigned int count) const;

    //TrackletEngine displaced name
    std::string TEDName(unsigned int l1,
                        unsigned int ireg1,
                        unsigned int ivm1,
                        unsigned int l2,
                        unsigned int ireg2,
                        unsigned int ivm2,
                        unsigned int iseed) const;

    //Tracklet parameter memory name
    std::string TParName(unsigned int l1, unsigned int l2, unsigned int l3, unsigned int itc) const;

    //TrackletCalculator displaced name
    std::string TCDName(unsigned int l1, unsigned int l2, unsigned int l3, unsigned int itc) const;

    //TrackletProjection name
    std::string TPROJName(unsigned int l1,
                          unsigned int l2,
                          unsigned int l3,
                          unsigned int itc,
                          unsigned int projlay,
                          unsigned int projreg) const;

    //FitTrack module name
    std::string FTName(unsigned int l1, unsigned int l2, unsigned int l3) const;

    //
    // This group of methods are used to print out the configuration as a file
    //
    void writeProjectionMemories(std::ostream& os, std::ostream& memories, std::ostream& modules);

    void writeSPMemories(std::ostream& os, std::ostream& memories, std::ostream& modules);

    void writeSPDMemories(std::ostream& os, std::ostream& memories, std::ostream& modules);

    void writeAPMemories(std::ostream& os, std::ostream& memories, std::ostream& modules);

    void writeCMMemories(std::ostream& os, std::ostream& memories, std::ostream& modules);

    void writeVMPROJMemories(std::ostream& os, std::ostream& memories, std::ostream& modules);

    void writeFMMemories(std::ostream& os, std::ostream& memories, std::ostream& modules);

    void writeASMemories(std::ostream& os, std::ostream& memories, std::ostream& modules);

    void writeVMSMemories(std::ostream& os, std::ostream& memories, std::ostream& modules);

    void writeTPARMemories(std::ostream& os, std::ostream& memories, std::ostream& modules);

    void writeTFMemories(std::ostream& os, std::ostream& memories, std::ostream& modules);

    void writeCTMemories(std::ostream& os, std::ostream& memories, std::ostream& modules);

    void writeILMemories(std::ostream& os, std::ostream& memories, std::ostream& modules);

    //
    //--- Store constants extracted from Settings
    //

    unsigned int NSector_;  //Number of sectors
    double rcrit_;          //critical radius that defines the sector

    bool combinedmodules_;  //if true write configuration for combined modules

    bool extended_;  //if true write configuration for extended configuration

    double rinvmax_;         //Max value for valid rinv
    double rmaxdisk_;        //Maximim disk radius
    double zlength_;         //Maximim (abslute) z-positon in barrel
    double rmean_[N_LAYER];  //Mean layer radius
    double zmean_[N_DISK];   //Mean disk z-position

    double dphisectorHG_;  //Full sector width

    unsigned int NTC_[N_SEED_PROMPT];  //Number of TC per seeding combination

    unsigned int NRegions_[N_LAYER + N_DISK];                     //Regions (all stubs memories 6 layers +5 disks
    unsigned int NVMME_[N_LAYER + N_DISK];                        //Number of MEs (all stubs memories 6 layers +5 disks
    std::pair<unsigned int, unsigned int> NVMTE_[N_SEED_PROMPT];  //number of TEs for each seeding combination

    //Min and max phi for a phi region (e.g. all stubs)
    std::vector<std::pair<double, double> > allStubs_[N_LAYER + N_DISK];

    //Min and max phi for VM bin
    std::vector<std::pair<double, double> > VMStubsME_[N_LAYER + N_DISK];

    //Phi ranges for the stubs in the VM bins used in the pair in th TE
    std::pair<std::vector<std::pair<double, double> >, std::vector<std::pair<double, double> > >
        VMStubsTE_[N_SEED_PROMPT];

    // VM bin in inner/outer seeding layer of each TE.
    std::vector<std::pair<unsigned int, unsigned int> > TE_[N_SEED_PROMPT];

    //The ID of all TE that send data to TCs for each seeding combination
    std::vector<std::vector<unsigned int> > TC_[N_SEED_PROMPT];

    //The projections to each layer/disk from a seed and TC
    std::vector<std::vector<std::pair<unsigned int, unsigned int> > > projections_[N_LAYER + N_DISK];

    //Which matches are used for each seeding layer
    //                                                L1 L2 L3 L4 L5 L6 D1 D2 D3 D4 D5
    int matchport_[N_SEED_PROMPT][N_LAYER + N_DISK] = {{-1, -1, 1, 2, 3, 4, 4, 3, 2, 1, -1},       //L1L2
                                                       {1, -1, -1, 2, 3, -1, 4, 3, 2, 1, -1},      //L2L3
                                                       {1, 2, -1, -1, 3, 4, 4, 3, -1, -1, -1},     //L3L4
                                                       {1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1},   //L5L6
                                                       {1, 2, -1, -1, -1, -1, -1, -1, 2, 3, 4},    //D1D2
                                                       {1, -1, -1, -1, -1, -1, 2, 3, -1, -1, 4},   //D3D4
                                                       {-1, -1, -1, -1, -1, -1, -1, 1, 2, 3, 4},   //L1D1
                                                       {1, -1, -1, -1, -1, -1, -1, 2, 3, 4, -1}};  //L2D1

    struct DTCinfo {
      std::string name;
      int layer;
      float phimin;
      float phimax;
    };
    std::list<DTCinfo> vecDTCinfo_;

    //Settings
    const Settings& settings_;
  };
}  // namespace trklet
#endif
