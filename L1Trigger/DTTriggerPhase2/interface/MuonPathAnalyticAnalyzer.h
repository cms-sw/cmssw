#ifndef L1Trigger_DTTriggerPhase2_MuonPathAnalyticAnalyzer_h
#define L1Trigger_DTTriggerPhase2_MuonPathAnalyticAnalyzer_h

#include "L1Trigger/DTTriggerPhase2/interface/MuonPathAnalyzer.h"

// ===============================================================================
// Previous definitions and declarations
// ===============================================================================

struct MAGNITUDE {
  int add;
  int coeff[4];
  int mult;
};

struct CONSTANTS {
  MAGNITUDE pos;
  MAGNITUDE slope;
  MAGNITUDE slope_xhh;
  MAGNITUDE t0;
};

struct LATCOMB_CONSTANTS {
  int latcomb;
  CONSTANTS constants;
};

struct CELL_VALID_LAYOUT {
  int cell_horiz_layout[4];
  int valid[4];
};

struct CELL_VALID_LAYOUT_CONSTANTS {
  CELL_VALID_LAYOUT cell_valid_layout;
  LATCOMB_CONSTANTS latcomb_constants[6];
};

// ===============================================================================
// Class declarations
// ===============================================================================

class MuonPathAnalyticAnalyzer : public MuonPathAnalyzer {
public:
  // Constructors and destructor
  MuonPathAnalyticAnalyzer(const edm::ParameterSet &pset,
                           edm::ConsumesCollector &iC,
                           std::shared_ptr<GlobalCoordsObtainer> &globalcoordsobtainer);
  ~MuonPathAnalyticAnalyzer() override;

  // Main methods
  void initialise(const edm::EventSetup &iEventSetup) override;
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           MuonPathPtrs &inMpath,
           std::vector<cmsdt::metaPrimitive> &metaPrimitives) override;
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           MuonPathPtrs &inMpath,
           MuonPathPtrs &outMPath) override{};

  void finish() override;

  // Other public methods

  bool hasPosRF(int wh, int sec) { return wh > 0 || (wh == 0 && sec % 4 > 1); };

  // Public attributes
  DTGeometry const *dtGeo_;
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomH;

  //shift
  edm::FileInPath shift_filename_;
  std::map<int, float> shiftinfo_;

  //shift theta
  edm::FileInPath shift_theta_filename_;
  std::map<int, float> shiftthetainfo_;

  int chosen_sl_;

private:
  // Private methods
  void analyze(MuonPathPtr &inMPath, std::vector<cmsdt::metaPrimitive> &metaPrimitives);
  void fillLAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER();
  void segment_fitter(DTSuperLayerId MuonPathSLId,
                      int wires[4],
                      int t0s[4],
                      int valid[4],
                      int reduced_times[4],
                      int cell_horiz_layout[4],
                      LATCOMB_CONSTANTS latcomb_consts,
                      int xwire_mm[4],
                      int coarse_pos,
                      int coarse_offset,
                      std::vector<cmsdt::metaPrimitive> &metaPrimitives);
  int compute_parameter(MAGNITUDE constants, int t0s[4], int DIV_SHR_BITS, int INCREASED_RES);
  std::vector<int> getLateralityCombination(int latcomb);

  // Private attributes

  const bool debug_;
  double chi2Th_;
  double tanPhiTh_;
  double tanPhiThw2max_;
  double tanPhiThw2min_;
  double tanPhiThw1max_;
  double tanPhiThw1min_;
  double tanPhiThw0_;
  int cellLayout_[cmsdt::NUM_LAYERS];
  std::vector<CELL_VALID_LAYOUT_CONSTANTS> LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER;

  // global coordinates
  std::shared_ptr<GlobalCoordsObtainer> globalcoordsobtainer_;
};

#endif
