#ifndef L1Trigger_DTTriggerPhase2_MuonPathFitter_h
#define L1Trigger_DTTriggerPhase2_MuonPathFitter_h

#include "L1Trigger/DTTriggerPhase2/interface/MuonPathAnalyzer.h"
#include "L1Trigger/DTTriggerPhase2/interface/vhdl_functions.h"

// ===============================================================================
// Previous definitions and declarations
// ===============================================================================

using coeff_arr_t = std::vector<std::vector<int>>;
struct coeffs_t {
  coeff_arr_t t0;
  coeff_arr_t position;
  coeff_arr_t slope;
  coeffs_t()
      : t0(cmsdt::N_COEFFS, std::vector<int>(cmsdt::GENERIC_COEFF_WIDTH, 0)),
        position(cmsdt::N_COEFFS, std::vector<int>(cmsdt::GENERIC_COEFF_WIDTH, 0)),
        slope(cmsdt::N_COEFFS, std::vector<int>(cmsdt::GENERIC_COEFF_WIDTH, 0)) {}
};

struct SLhitP {
  int ti;  // unsigned(16 downto 0); -- 12 msb = bunch_ctr, 5 lsb = tdc counts, resolution 25/32 ns
  int wi;  // unsigned(6 downto 0); -- ~ 96 channels per layer
  int ly;  // unsigned(1 downto 0); -- 4 layers
  int wp;  // signed(WIREPOS_WIDTH-1 downto 0);
};

struct fit_common_in_t {
  // int valid; not needed, we will not propagate the mpath to the fitter
  std::vector<SLhitP> hits;
  std::vector<int> hits_valid;    // slv(0 to 7)
  std::vector<int> lateralities;  // slv(0 to 7)
  coeffs_t coeffs;
  int coarse_bctr;     // unsigned(11 downto 0)
  int coarse_wirepos;  // signed(WIDTH_FULL_POS-1 downto WIREPOS_NORM_LSB_IGNORED);
};

struct fit_common_out_t {
  int t0;
  int slope;
  int position;
  int chi2;
  int valid_fit;
  fit_common_out_t() : t0(0), slope(0), position(0), chi2(0), valid_fit(0) {}
};

// ===============================================================================
// Class declarations
// ===============================================================================

class MuonPathFitter : public MuonPathAnalyzer {
public:
  // Constructors and destructor
  MuonPathFitter(const edm::ParameterSet &pset,
                 edm::ConsumesCollector &iC,
                 std::shared_ptr<GlobalCoordsObtainer> &globalcoordsobtainer);
  ~MuonPathFitter() override;

  // Main methods

  // Other public methods
  coeffs_t RomDataConvert(std::vector<int> slv,
                          short COEFF_WIDTH_T0,
                          short COEFF_WIDTH_POSITION,
                          short COEFF_WIDTH_SLOPE,
                          short LOLY,
                          short HILY);

  bool hasPosRF(int wh, int sec) { return wh > 0 || (wh == 0 && sec % 4 > 1); };
  void setChi2Th(double chi2Th) { chi2Th_ = chi2Th; };
  void setTanPhiTh(double tanPhiTh) { tanPhiTh_ = tanPhiTh; };

  // Public attributes
  DTGeometry const *dtGeo_;
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomH;

  //shift
  edm::FileInPath shift_filename_;
  std::map<int, float> shiftinfo_;

  // max drift velocity
  edm::FileInPath maxdrift_filename_;
  int maxdriftinfo_[5][4][14];
  int max_drift_tdc = -1;

  int get_rom_addr(MuonPathPtr &inMPath, latcomb lats);
  fit_common_out_t fit(fit_common_in_t fit_common_in,
                       int XI_WIDTH,
                       int COEFF_WIDTH_T0,
                       int COEFF_WIDTH_POSITION,
                       int COEFF_WIDTH_SLOPE,
                       int PRECISSION_T0,
                       int PRECISSION_POSITION,
                       int PRECISSION_SLOPE,
                       int PROD_RESIZE_T0,
                       int PROD_RESIZE_POSITION,
                       int PROD_RESIZE_SLOPE,
                       int MAX_DRIFT_TDC,
                       int sl);

  double tanPhiTh_;
  const bool debug_;
  double chi2Th_;

  // global coordinates
  std::shared_ptr<GlobalCoordsObtainer> globalcoordsobtainer_;

private:
  // Private methods

  // Private attributes
};

#endif
