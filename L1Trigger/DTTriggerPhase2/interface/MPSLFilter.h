#ifndef Phase2L1Trigger_DTTrigger_MPSLFilter_h
#define Phase2L1Trigger_DTTrigger_MPSLFilter_h

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

struct valid_tp_t {
  bool valid;
  cmsdt::metaPrimitive mp;
  valid_tp_t() : valid(false), mp(cmsdt::metaPrimitive()) {}
  valid_tp_t(bool valid, cmsdt::metaPrimitive mp) : valid(valid), mp(mp) {}
};

using valid_tp_arr_t = std::vector<valid_tp_t>;

class MPSLFilter : public MPFilter {
public:
  // Constructors and destructor
  MPSLFilter(const edm::ParameterSet &pset);
  ~MPSLFilter() override = default;

  // Main methods
  void initialise(const edm::EventSetup &iEventSetup) override;
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           std::vector<cmsdt::metaPrimitive> &inMPath,
           std::vector<cmsdt::metaPrimitive> &outMPath) override;
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           std::vector<cmsdt::metaPrimitive> &inSLMPath,
           std::vector<cmsdt::metaPrimitive> &inCorMPath,
           std::vector<cmsdt::metaPrimitive> &outMPath) override{};
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
  // std::vector<cmsdt::metaPrimitive> filter(std::map<int, std::vector<cmsdt::metaPrimitive>>);
  std::vector<cmsdt::metaPrimitive> filter(std::vector<cmsdt::metaPrimitive> mps);
  bool isDead(cmsdt::metaPrimitive mp, std::map<int, valid_tp_arr_t> tps_per_bx);
  int killTps(cmsdt::metaPrimitive mp, int bx, std::map<int, valid_tp_arr_t> &tps_per_bx);
  int share_hit(cmsdt::metaPrimitive mp, cmsdt::metaPrimitive mp2);
  int match(cmsdt::metaPrimitive mp1, cmsdt::metaPrimitive mp2);
  int smaller_chi2(cmsdt::metaPrimitive mp, cmsdt::metaPrimitive mp2);
  int get_chi2(cmsdt::metaPrimitive mp);

  // Private attributes
  const bool debug_;
};

#endif
