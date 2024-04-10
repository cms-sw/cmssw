#ifndef Phase2L1Trigger_DTTrigger_MPCorFilter_h
#define Phase2L1Trigger_DTTrigger_MPCorFilter_h

#include "L1Trigger/DTTriggerPhase2/interface/MPFilter.h"
#include "L1Trigger/DTTriggerPhase2/interface/vhdl_functions.h"

#include <iostream>
#include <fstream>

// ===============================================================================
// Previous definitions and declarations
// ===============================================================================

// ===============================================================================
// Class declarations
// ===============================================================================

struct valid_cor_tp_t {
  bool valid;
  cmsdt::metaPrimitive mp;
  int coarsed_t0;
  int coarsed_pos;
  int coarsed_slope;
  valid_cor_tp_t() : valid(false), mp(cmsdt::metaPrimitive()), coarsed_t0(-1), coarsed_pos(-1), coarsed_slope(-1) {}
  valid_cor_tp_t(bool valid, cmsdt::metaPrimitive mp, int coarsed_t0, int coarsed_pos, int coarsed_slope)
      : valid(valid), mp(mp), coarsed_t0(coarsed_t0), coarsed_pos(coarsed_pos), coarsed_slope(coarsed_slope) {}
};

using valid_cor_tp_arr_t = std::vector<valid_cor_tp_t>;

class MPCorFilter : public MPFilter {
public:
  // Constructors and destructor
  MPCorFilter(const edm::ParameterSet &pset);
  ~MPCorFilter() override = default;

  // Main methods
  void initialise(const edm::EventSetup &iEventSetup) override;
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           std::vector<cmsdt::metaPrimitive> &inMPath,
           std::vector<cmsdt::metaPrimitive> &outMPath) override{};
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           std::vector<cmsdt::metaPrimitive> &inSLMPath,
           std::vector<cmsdt::metaPrimitive> &inCorMPath,
           std::vector<cmsdt::metaPrimitive> &outMPath) override;
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           MuonPathPtrs &inMPath,
           MuonPathPtrs &outMPath) override{};

  void finish() override;

  // Other public methods

  // Public attributes
  void printmP(cmsdt::metaPrimitive mP);

private:
  // Private methods
  std::vector<cmsdt::metaPrimitive> filter(std::vector<cmsdt::metaPrimitive> SL1mps,
                                           std::vector<cmsdt::metaPrimitive> SL2mps,
                                           std::vector<cmsdt::metaPrimitive> SL3mps,
                                           std::vector<cmsdt::metaPrimitive> Cormps);
  std::vector<int> coarsify(cmsdt::metaPrimitive mp, int sl);
  bool isDead(cmsdt::metaPrimitive mp, std::vector<int> coarsed, std::map<int, valid_cor_tp_arr_t> tps_per_bx);
  int killTps(cmsdt::metaPrimitive mp, std::vector<int> coarsed, int bx, std::map<int, valid_cor_tp_arr_t> &tps_per_bx);
  int match(cmsdt::metaPrimitive mp, std::vector<int> coarsed, valid_cor_tp_t valid_cor_tp2);
  int get_chi2(cmsdt::metaPrimitive mp);

  // Private attributes
  const bool debug_;
};

#endif
